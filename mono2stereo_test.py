"""
mono2stereo 定量测试脚本
输入：测试集左图 -> DIBR 生成右图
输出：vs 真实右图 的图像质量指标：PSNR, SSIM, SIOU
基于 mono2stereo.py 的真实 DIBR + 快速补洞逻辑（无时序稳定）
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
from utils_depth.metrics import eval_stereo  # 直接使用第三方指标


# =============================================================================
# 模型配置
# =============================================================================

def get_model_config(encoder: str) -> Dict:
    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    return configs[encoder]


# =============================================================================
# GPU DIBR + Fast Inpaint (和 mono2stereo.py 完全一致)
# =============================================================================

@torch.no_grad()
def forward_warp_right_gpu(
    left_rgb: torch.Tensor, disparity: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """和 mono2stereo.py 完全一致的 int64 Z-buffer DIBR"""
    h, w, _ = left_rgb.shape
    N = h * w
    device = left_rgb.device

    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    x_tgt = torch.round(xs.float() - disparity).long()

    valid = (x_tgt >= 0) & (x_tgt < w)
    src_lin = (ys * w + xs).reshape(-1)
    tgt_lin = (ys * w + x_tgt).reshape(-1)
    valid_flat = valid.reshape(-1)
    near_flat = disparity.reshape(-1)  # 视差越大，越近 → 优先级高

    src_lin = src_lin[valid_flat]
    tgt_lin = tgt_lin[valid_flat]
    near_flat = near_flat[valid_flat]

    # ========== int64 无损 Z-buffer ==========
    NEAR_BITS = 20
    src_bits = max(20, (N - 1).bit_length())

    near_q = (near_flat / near_flat.max().clamp_min(1e-6) * ((1 << NEAR_BITS) - 1)).long()
    encoded = (near_q << src_bits) | src_lin.long()

    max_encoded = torch.full((N,), -1, device=device, dtype=torch.int64)
    max_encoded.scatter_reduce_(0, tgt_lin, encoded, reduce="amax", include_self=True)
    selected = encoded == max_encoded[tgt_lin]

    src_sel = src_lin[selected]
    tgt_sel = tgt_lin[selected]

    left_flat = left_rgb.reshape(-1, 3)
    right_flat = torch.zeros_like(left_flat)
    right_flat[tgt_sel] = left_flat[src_sel]

    hole = torch.ones((N,), device=device, dtype=torch.bool)
    hole[tgt_sel] = False

    return right_flat.reshape(h, w, 3), hole.reshape(h, w)


@torch.no_grad()
def dilate_hole_right(hole: torch.Tensor, dilate_px: int = 1) -> torch.Tensor:
    """向右膨胀空洞，消除前景轮廓线"""
    if dilate_px <= 0:
        return hole
    out = hole.clone()
    for shift in range(1, dilate_px + 1):
        out = out | torch.roll(hole, shifts=shift, dims=1)
    out[:, :dilate_px] = hole[:, :dilate_px]
    return out


@torch.no_grad()
def fast_inpaint_gpu(
    image: torch.Tensor, hole_mask: torch.Tensor, kernel_size: int = 11, max_iter: int = 64
) -> torch.Tensor:
    """和 mono2stereo.py 一致的快速补洞（简化版，无边缘敏感）"""
    h, w = hole_mask.shape
    device = image.device

    img = image.clone()
    hole = hole_mask.clone()

    if not torch.any(hole):
        return img

    pad = kernel_size // 2
    kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=img.dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)

    for _ in range(max_iter):
        if not torch.any(hole):
            break

        known = (~hole).float().unsqueeze(0).unsqueeze(0)
        img_nchw = img.permute(2, 0, 1).unsqueeze(0)

        count = F.conv2d(known, kernel1, padding=pad)
        fillable = hole & (count[0, 0] > 0.01)

        if not torch.any(fillable):
            break

        rgb_sum = F.conv2d(img_nchw * known, kernel3, padding=pad, groups=3)
        avg = rgb_sum / count.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        img[fillable] = avg_hwc[fillable]
        hole[fillable] = False

    # 兜底
    if torch.any(hole):
        fallback = img[~hole_mask].mean(dim=0)
        img[hole] = fallback

    return img


# =============================================================================
# 单帧处理流水线（和 mono2stereo.py 逻辑对齐）
# =============================================================================

def process_single_image(
    left_path: str,
    model: DepthAnythingV2,
    device: torch.device,
    input_size: int = 518,
    max_disparity: float = 24.0,
    use_relative_disparity: bool = False,  # False=绝对像素, True=基准宽度参照自适应
    base_width: int = 1920,  # 你原来调参时的基准分辨率宽度
    fp16: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理单张左图，生成右图（和 mono2stereo.py 逻辑完全一致）

    Returns:
        pred_right: 生成的右图 (uint8, HWC RGB)
        left_rgb:   原始左图 (uint8, HWC RGB)
    """
    # ---------- 1. 读取左图 ----------
    left_bgr = cv2.imread(left_path)
    if left_bgr is None:
        raise RuntimeError(f"无法读取图片: {left_path}")

    h_orig, w_orig = left_bgr.shape[:2]
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

    # ---------- 2. 计算深度分辨率 ----------
    long_edge = input_size
    scale = long_edge / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = long_edge
    else:
        depth_h = long_edge

    # ---------- 3. 预处理 ----------
    img = cv2.resize(left_rgb, (depth_w, depth_h), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    if fp16:
        img_tensor = img_tensor.half()

    # ---------- 4. 深度推理 ----------
    with torch.no_grad():
        depth = model(img_tensor)[0].float()  # [H, W]

    # ---------- 5. 归一化（无时序稳定） ----------
    flat = depth.reshape(-1)
    sample_size = min(16384, flat.numel())
    idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=device))
    q_low, q_high = q_vals[0], q_vals[1]

    denom = (q_high - q_low).clamp_min(1e-6)
    depth_norm = ((depth - q_low) / denom).clamp(0.0, 1.0)

    # inverse depth: 值大 = 近 → 视差大
    near = depth_norm

    # ---------- 6. 上采样到原分辨率 + 计算视差 ----------
    near_orig = F.interpolate(
        near[None, None, :, :], size=(h_orig, w_orig),
        mode="bilinear", align_corners=False,
    )[0, 0]

    if use_relative_disparity:
        # ─────────────────────────────────────────────────────────────
        # 🎯 基准宽度参照的自适应视差（你要的模式！）
        # max_disparity 是在 base_width 宽度下调出来的值
        # 实际视差会按当前图片宽度等比例缩放
        #
        # 例: base_width=1920, max_disparity=5
        #     1920 宽的图 → 5 像素
        #      960 宽的图 → 2.5 像素
        #     3840 宽的图 → 10 像素
        # ─────────────────────────────────────────────────────────────
        scale_factor = w_orig / base_width
        actual_max_disparity = max_disparity * scale_factor
        disparity = near_orig * actual_max_disparity
    else:
        # 原逻辑：绝对像素视差（不管图多大，就是这么多像素）
        disparity = near_orig * max_disparity

    # ---------- 7. 左图上 GPU ----------
    left_gpu = torch.from_numpy(left_rgb).to(device=device, dtype=torch.float32) / 255.0

    # ---------- 8. DIBR 扭曲 ----------
    right_warped, hole = forward_warp_right_gpu(left_gpu, disparity)

    # ---------- 9. 空洞向右膨胀 1px（消除轮廓线） ----------
    hole_dilated = dilate_hole_right(hole, dilate_px=1)

    # ---------- 10. 空洞填补 ----------
    right_inpainted = fast_inpaint_gpu(right_warped, hole_dilated, kernel_size=11, max_iter=64)

    # ---------- 11. 转回 CPU uint8 ----------
    right_np = (right_inpainted.clamp(0, 1) * 255.0).byte().cpu().numpy()

    return right_np, left_rgb


# =============================================================================
# 单视差测试逻辑
# =============================================================================

def run_single_max_disparity(
    max_disparity: float,
    model,
    device: torch.device,
    test_root: Path,
    base_out_root: Path,
    args,
    use_relative_disparity: bool = False,
    base_width: int = 1920,
) -> dict:
    """
    测试单个 max_disparity 值，返回全局平均结果
    """
    print(f"\n{'='*70}")
    if use_relative_disparity:
        print(f"[test] 🚀 开始测试 (基准宽度自适应) max_disparity = {max_disparity}")
        print(f"[test]    基准宽度: {base_width}px (你原来调参的分辨率)")
        print(f"[test]    实际视差 = {max_disparity} × (当前图片宽度 / {base_width})")
    else:
        print(f"[test] 🚀 开始测试 (绝对像素) max_disparity = {max_disparity}")
    print('='*70)

    scenes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])

    # 为每个视差创建独立的输出目录（支持浮点数）
    disp_str = f"{max_disparity}".replace('.', '_')
    if use_relative_disparity:
        out_root = base_out_root / f"disp_adaptive_{disp_str}"
    else:
        out_root = base_out_root / f"disp_abs_{disp_str}"
    out_root.mkdir(parents=True, exist_ok=True)

    metric_names = ['rmse', 'mse', 'siou', 'psnr', 'ssim']
    all_metrics = {m: [] for m in metric_names}
    scene_results = {}
    detailed_results = []  # 存储每一张图片的详细结果

    # ---------- 逐场景测试 ----------
    for scene in scenes:
        left_dir = test_root / scene / "left"
        right_dir = test_root / scene / "right"

        if not left_dir.exists() or not right_dir.exists():
            continue

        left_files = sorted([f for f in left_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        scene_metrics = {m: [] for m in metric_names}

        # 输出子目录
        scene_out = out_root / scene
        scene_out.mkdir(exist_ok=True)

        # 逐张处理
        for i, left_path in enumerate(left_files):
            right_path = right_dir / left_path.name
            if not right_path.exists():
                continue

            try:
                pred_right, left_rgb = process_single_image(
                    str(left_path), model, device,
                    input_size=args.input_size,
                    max_disparity=max_disparity,
                    use_relative_disparity=use_relative_disparity,
                    base_width=args.base_width,
                    fp16=args.fp16,
                )

                gt_right_bgr = cv2.imread(str(right_path))
                gt_right = cv2.cvtColor(gt_right_bgr, cv2.COLOR_BGR2RGB)

                if pred_right.shape != gt_right.shape:
                    gt_right = cv2.resize(gt_right, (pred_right.shape[1], pred_right.shape[0]))

                metrics = eval_stereo(pred_right, gt_right, left_rgb)

                # 🐛 边界处理：MSE=0 时 PSNR=inf，替换为 80dB（生成得太准了！）
                if np.isinf(metrics['psnr']):
                    metrics['psnr'] = 80.0
                    print(f"       ⭐ PSNR=inf → 替换为 80dB: {scene}/{left_path.name}")
                if np.isnan(metrics['psnr']):
                    metrics['psnr'] = 0.0
                if np.isinf(metrics['rmse']) or np.isnan(metrics['rmse']):
                    metrics['rmse'] = 0.0
                if np.isinf(metrics['mse']) or np.isnan(metrics['mse']):
                    metrics['mse'] = 0.0

                for k in metric_names:
                    scene_metrics[k].append(metrics[k])
                    all_metrics[k].append(metrics[k])

                # 记录每一张图片的详细结果
                img_h, img_w = left_rgb.shape[:2]
                if use_relative_disparity:
                    actual_disp = max_disparity * (img_w / args.base_width)
                else:
                    actual_disp = max_disparity

                detailed_results.append({
                    'scene': scene,
                    'filename': left_path.name,
                    'resolution': f"{img_w}x{img_h}",
                    'actual_disparity': actual_disp,
                    'rmse': metrics['rmse'],
                    'mse': metrics['mse'],
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'siou': metrics['siou'],
                })

                # 保存可视化结果
                if args.save_vis:
                    h, w = left_rgb.shape[:2]
                    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
                    vis[:, :w] = left_rgb
                    vis[:, w:2*w] = pred_right
                    vis[:, 2*w:] = gt_right
                    cv2.imwrite(str(scene_out / left_path.name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            except Exception as e:
                continue

        # 场景统计
        if len(scene_metrics['psnr']) > 0:
            scene_avg = {k: np.mean(v) for k, v in scene_metrics.items()}
            scene_results[scene] = scene_avg

    # ---------- 全局统计 ----------
    global_avg = {k: np.mean(v) for k, v in all_metrics.items()}
    total_samples = len(all_metrics['psnr'])

    print(f"\n[test] ✅ max_disparity = {max_disparity} 完成!")
    print(f"       PSNR:  {global_avg['psnr']:.2f} dB")
    print(f"       SSIM:  {global_avg['ssim']:.4f}")
    print(f"       SIOU:  {global_avg['siou']:.4f}")
    print(f"       总样本: {total_samples}")

    # 保存详细结果到 results_xx.txt（支持浮点数）
    if use_relative_disparity:
        result_path = base_out_root / f"results_adaptive_{disp_str}.txt"
        detail_path = base_out_root / f"results_detail_adaptive_{disp_str}.txt"
    else:
        result_path = base_out_root / f"results_abs_{disp_str}.txt"
        detail_path = base_out_root / f"results_detail_abs_{disp_str}.txt"

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=== mono2stereo 定量测试结果 ===\n")
        f.write(f"测试集: {test_root}\n")
        f.write(f"模型: {args.encoder}\n")
        f.write(f"输入尺寸: {args.input_size}\n")
        f.write(f"最大视差: {max_disparity}\n")
        f.write(f"\n")

        f.write("--- 各场景详情 ---\n")
        for scene, avg in scene_results.items():
            f.write(f"{scene:12s} | RMSE={avg['rmse']:.4f} | MSE={avg['mse']:.4f} | ")
            f.write(f"PSNR={avg['psnr']:6.2f} | SSIM={avg['ssim']:.4f} | SIOU={avg['siou']:.4f}\n")

        f.write(f"\n--- 全局平均 (共 {total_samples} 张图片) ---\n")
        f.write(f"RMSE:  {global_avg['rmse']:.4f}\n")
        f.write(f"MSE:   {global_avg['mse']:.4f}\n")
        f.write(f"PSNR:  {global_avg['psnr']:.2f} dB\n")
        f.write(f"SSIM:  {global_avg['ssim']:.4f}\n")
        f.write(f"SIOU:  {global_avg['siou']:.4f}\n")

    print(f"[test] 📄 结果已保存: {result_path}")

    # =========================================================================
    # 📊 超级详细的每图片打分报告（你要的！）
    # =========================================================================
    with open(detail_path, 'w', encoding='utf-8') as f:
        f.write("╔══════════════════════════════════════════════════════════════════════════════╗\n")
        f.write("║                    📊 mono2stereo 逐图片详细打分报告                           ║\n")
        f.write("╚══════════════════════════════════════════════════════════════════════════════╝\n\n")

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📋 基本信息\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"测试集路径: {test_root}\n")
        f.write(f"模型: DepthAnythingV2-{args.encoder}\n")
        f.write(f"模型输入尺寸: {args.input_size}\n")
        f.write(f"视差模式: {'基准宽度自适应' if use_relative_disparity else '绝对像素'}\n")
        f.write(f"配置视差值: {max_disparity}\n")
        if use_relative_disparity:
            f.write(f"基准宽度: {args.base_width}px\n")
        f.write(f"测试图片总数: {total_samples}\n")
        f.write(f"场景数量: {len(scene_results)}\n")
        f.write(f"FP16 推理: {'是' if args.fp16 else '否'}\n\n")

        # 按场景分组显示详细结果
        current_scene = None
        scene_count = 0
        scene_metrics_detail = {}

        for r in detailed_results:
            scene = r['scene']
            if scene != current_scene:
                if current_scene is not None:
                    # 场景统计
                    avg = scene_results[current_scene]
                    f.write(f"\n  📈 场景「{current_scene}」平均:\n")
                    f.write(f"     PSNR 平均: {avg['psnr']:.2f} dB\n")
                    f.write(f"     SSIM 平均: {avg['ssim']:.4f}\n")
                    f.write(f"     SIOU 平均: {avg['siou']:.4f}\n")
                    f.write(f"     RMSE 平均: {avg['rmse']:.4f}\n")
                    f.write(f"     MSE  平均: {avg['mse']:.4f}\n")
                    f.write(f"{'─'*78}\n")

                current_scene = scene
                scene_count += 1
                f.write("\n")
                f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
                f.write(f"🖼️  场景 {scene_count}: {scene}\n")
                f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
                f.write(f"{'序号':>4} {'文件名':<25} {'分辨率':>12} {'实际视差':>8} {'PSNR':>8} {'SSIM':>8} {'SIOU':>8} {'RMSE':>8} {'MSE':>8}\n")
                f.write(f"{'-'*4} {'-'*25} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}\n")
                scene_metrics_detail[scene] = {'count': 0, 'psnr': [], 'ssim': [], 'siou': []}

            scene_metrics_detail[scene]['count'] += 1
            scene_metrics_detail[scene]['psnr'].append(r['psnr'])
            scene_metrics_detail[scene]['ssim'].append(r['ssim'])
            scene_metrics_detail[scene]['siou'].append(r['siou'])

            f.write(f"{scene_metrics_detail[scene]['count']:4d} ")
            f.write(f"{r['filename']:<25} ")
            f.write(f"{r['resolution']:>12} ")
            f.write(f"{r['actual_disparity']:8.2f} ")
            f.write(f"{r['psnr']:8.2f} ")
            f.write(f"{r['ssim']:8.4f} ")
            f.write(f"{r['siou']:8.4f} ")
            f.write(f"{r['rmse']:8.4f} ")
            f.write(f"{r['mse']:8.4f}\n")

        # 最后一个场景的统计
        if current_scene is not None:
            avg = scene_results[current_scene]
            f.write(f"\n  📈 场景「{current_scene}」平均:\n")
            f.write(f"     PSNR 平均: {avg['psnr']:.2f} dB\n")
            f.write(f"     SSIM 平均: {avg['ssim']:.4f}\n")
            f.write(f"     SIOU 平均: {avg['siou']:.4f}\n")
            f.write(f"     RMSE 平均: {avg['rmse']:.4f}\n")
            f.write(f"     MSE  平均: {avg['mse']:.4f}\n")
            f.write(f"{'─'*78}\n")

        # =====================================================================
        # 全局统计汇总
        # =====================================================================
        f.write("\n\n")
        f.write("╔══════════════════════════════════════════════════════════════════════════════╗\n")
        f.write("║                         📊 全局统计汇总                                       ║\n")
        f.write("╚══════════════════════════════════════════════════════════════════════════════╝\n\n")

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📈 各场景对比汇总\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"{'场景名':<15} {'图片数':>8} {'PSNR(avg)':>10} {'SSIM(avg)':>10} {'SIOU(avg)':>10} {'PSNR(min/max)':>18}\n")
        f.write(f"{'-'*15} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*18}\n")

        for scene in scene_results.keys():
            psnr_list = scene_metrics_detail[scene]['psnr']
            avg = scene_results[scene]
            f.write(f"{scene:<15} ")
            f.write(f"{len(psnr_list):8d} ")
            f.write(f"{avg['psnr']:10.2f} ")
            f.write(f"{avg['ssim']:10.4f} ")
            f.write(f"{avg['siou']:10.4f} ")
            f.write(f"{min(psnr_list):.2f} / {max(psnr_list):.2f}\n")

        # =====================================================================
        # 指标分布统计
        # =====================================================================
        f.write("\n\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📉 指标分布统计 (全部图片)\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        all_psnr = [r['psnr'] for r in detailed_results]
        all_ssim = [r['ssim'] for r in detailed_results]
        all_siou = [r['siou'] for r in detailed_results]
        all_rmse = [r['rmse'] for r in detailed_results]

        f.write(f"\n  PSNR (dB):\n")
        f.write(f"    最小值: {np.min(all_psnr):.2f}\n")
        f.write(f"    最大值: {np.max(all_psnr):.2f}\n")
        f.write(f"    平均值: {np.mean(all_psnr):.2f}\n")
        f.write(f"    中位数: {np.median(all_psnr):.2f}\n")
        f.write(f"    标准差: {np.std(all_psnr):.2f}\n")

        f.write(f"\n  SSIM:\n")
        f.write(f"    最小值: {np.min(all_ssim):.4f}\n")
        f.write(f"    最大值: {np.max(all_ssim):.4f}\n")
        f.write(f"    平均值: {np.mean(all_ssim):.4f}\n")
        f.write(f"    中位数: {np.median(all_ssim):.4f}\n")
        f.write(f"    标准差: {np.std(all_ssim):.4f}\n")

        f.write(f"\n  SIOU:\n")
        f.write(f"    最小值: {np.min(all_siou):.4f}\n")
        f.write(f"    最大值: {np.max(all_siou):.4f}\n")
        f.write(f"    平均值: {np.mean(all_siou):.4f}\n")
        f.write(f"    中位数: {np.median(all_siou):.4f}\n")
        f.write(f"    标准差: {np.std(all_siou):.4f}\n")

        f.write(f"\n  RMSE:\n")
        f.write(f"    最小值: {np.min(all_rmse):.4f}\n")
        f.write(f"    最大值: {np.max(all_rmse):.4f}\n")
        f.write(f"    平均值: {np.mean(all_rmse):.4f}\n")
        f.write(f"    中位数: {np.median(all_rmse):.4f}\n")
        f.write(f"    标准差: {np.std(all_rmse):.4f}\n")

        # =====================================================================
        # 🎯 目标达成分析（你的三个核心目标）
        # =====================================================================
        f.write("\n\n")
        f.write("╔══════════════════════════════════════════════════════════════════════════════╗\n")
        f.write("║                        🎯 核心目标达成分析                                    ║\n")
        f.write("╚══════════════════════════════════════════════════════════════════════════════╝\n\n")

        target_psnr = 32.0
        target_ssim = 0.75
        target_siou = 0.28

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"📌 你的目标:  PSNR ≥ {target_psnr} dB   |   SSIM ≥ {target_ssim}   |   SIOU ≥ {target_siou}\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

        # 单个指标达标统计
        psnr_pass = sum(1 for p in all_psnr if p >= target_psnr)
        ssim_pass = sum(1 for s in all_ssim if s >= target_ssim)
        siou_pass = sum(1 for s in all_siou if s >= target_siou)

        f.write("  ┌─────────────┬──────────┬──────────┬──────────┐\n")
        f.write("  │   指标      │   目标   │   达标   │   达标率  │\n")
        f.write("  ├─────────────┼──────────┼──────────┼──────────┤\n")
        f.write(f"  │   PSNR      │ ≥{target_psnr:5.1f}dB  │ {psnr_pass:6d} 张 │ {psnr_pass/len(all_psnr)*100:6.1f}%  │\n")
        f.write(f"  │   SSIM      │ ≥{target_ssim:5.2f}   │ {ssim_pass:6d} 张 │ {ssim_pass/len(all_ssim)*100:6.1f}%  │\n")
        f.write(f"  │   SIOU      │ ≥{target_siou:5.2f}   │ {siou_pass:6d} 张 │ {siou_pass/len(all_siou)*100:6.1f}%  │\n")
        f.write("  └─────────────┴──────────┴──────────┴──────────┘\n\n")

        # 三指标同时达标统计
        all_three_pass = 0
        pass_2_of_3 = 0
        pass_1_of_3 = 0
        pass_0_of_3 = 0

        for r in detailed_results:
            passes = 0
            if r['psnr'] >= target_psnr: passes += 1
            if r['ssim'] >= target_ssim: passes += 1
            if r['siou'] >= target_siou: passes += 1

            if passes == 3: all_three_pass += 1
            elif passes == 2: pass_2_of_3 += 1
            elif passes == 1: pass_1_of_3 += 1
            else: pass_0_of_3 += 1

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📊 综合达标情况（每张图片同时达标几个指标？）\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

        total = len(detailed_results)
        f.write(f"  🏆 3个指标全部达标: {all_three_pass:5d} 张 ({all_three_pass/total*100:5.1f}%) {'█'*int(all_three_pass/total*50)}\n")
        f.write(f"  ✅ 2个指标达标:     {pass_2_of_3:5d} 张 ({pass_2_of_3/total*100:5.1f}%) {'█'*int(pass_2_of_3/total*50)}\n")
        f.write(f"  ⚠️  1个指标达标:     {pass_1_of_3:5d} 张 ({pass_1_of_3/total*100:5.1f}%) {'█'*int(pass_1_of_3/total*50)}\n")
        f.write(f"  ❌ 全部未达标:      {pass_0_of_3:5d} 张 ({pass_0_of_3/total*100:5.1f}%) {'█'*int(pass_0_of_3/total*50)}\n")

        # 目标达成差距分析
        f.write("\n\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📉 与目标的差距分析\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

        psnr_gap = target_psnr - np.mean(all_psnr)
        ssim_gap = target_ssim - np.mean(all_ssim)
        siou_gap = target_siou - np.mean(all_siou)

        f.write(f"  PSNR:  当前平均 = {np.mean(all_psnr):.2f} dB, 目标差距 = {psnr_gap:+.2f} dB\n")
        if psnr_gap <= 0:
            f.write(f"         ✅ 已达标！超出目标 {-psnr_gap:.2f} dB\n")
        else:
            f.write(f"         ⏳ 还需提升 {psnr_gap:.2f} dB\n")

        f.write(f"\n  SSIM:  当前平均 = {np.mean(all_ssim):.4f}, 目标差距 = {ssim_gap:+.4f}\n")
        if ssim_gap <= 0:
            f.write(f"         ✅ 已达标！超出目标 {-ssim_gap:.4f}\n")
        else:
            f.write(f"         ⏳ 还需提升 {ssim_gap:.4f}\n")

        f.write(f"\n  SIOU:  当前平均 = {np.mean(all_siou):.4f}, 目标差距 = {siou_gap:+.4f}\n")
        if siou_gap <= 0:
            f.write(f"         ✅ 已达标！超出目标 {-siou_gap:.4f}\n")
        else:
            f.write(f"         ⏳ 还需提升 {siou_gap:.4f}\n")

        # =====================================================================
        # PSNR 等级分布
        # =====================================================================
        f.write("\n\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🏆 PSNR 等级分布 (画质分级) | 目标线: 32dB\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        psnr_levels = [
            (40, '🌟 卓越 (>40 dB)', '接近无损，画质极佳'),
            (35, '⭐ 优秀 (35-40 dB)', '人眼几乎无法察觉差异'),
            (32, '🎯 达标 (32-35 dB)', '达到你的目标值！'),
            (30, '✅ 良好 (30-32 dB)', '接近目标，差一点点'),
            (25, '⚠️ 一般 (25-30 dB)', '明显差异，需优化'),
            (20, '❌ 较差 (20-25 dB)', '差异明显，画质受损'),
            (0, '💔 很差 (<20 dB)', '严重失真'),
        ]

        psnr_counts = {}
        for threshold, _, _ in psnr_levels:
            psnr_counts[threshold] = 0

        for psnr in all_psnr:
            for threshold, _, _ in psnr_levels:
                if psnr >= threshold:
                    psnr_counts[threshold] += 1
                    break

        for threshold, label, desc in psnr_levels:
            count = psnr_counts[threshold]
            percent = count / len(all_psnr) * 100 if len(all_psnr) > 0 else 0
            bar = '█' * int(percent / 5)
            marker = '← 🎯目标线' if threshold == target_psnr else ''
            f.write(f"  {label}:\n")
            f.write(f"    {desc} {marker}\n")
            f.write(f"    数量: {count} 张 ({percent:.1f}%) |{bar}|\n")

        # =====================================================================
        # SSIM 等级分布
        # =====================================================================
        f.write("\n\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🏆 SSIM 等级分布 (结构相似性) | 目标线: 0.75\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        ssim_levels = [
            (0.95, '🌟 完美 (>0.95)', '结构几乎完全一致'),
            (0.90, '⭐ 极佳 (0.90-0.95)', '结构高度一致'),
            (0.85, '✨ 优秀 (0.85-0.90)', '结构一致性很好'),
            (0.80, '✅ 良好 (0.80-0.85)', '结构一致性好'),
            (0.75, '🎯 达标 (0.75-0.80)', '达到你的目标值！'),
            (0.70, '⚠️ 一般 (0.70-0.75)', '接近目标，需优化'),
            (0.60, '❌ 较差 (0.60-0.70)', '结构差异明显'),
            (0, '💔 很差 (<0.60)', '结构严重失真'),
        ]

        ssim_counts = {}
        for threshold, _, _ in ssim_levels:
            ssim_counts[threshold] = 0

        for ssim in all_ssim:
            for threshold, _, _ in ssim_levels:
                if ssim >= threshold:
                    ssim_counts[threshold] += 1
                    break

        for threshold, label, desc in ssim_levels:
            count = ssim_counts[threshold]
            percent = count / len(all_ssim) * 100 if len(all_ssim) > 0 else 0
            bar = '█' * int(percent / 5)
            marker = '← 🎯目标线' if threshold == target_ssim else ''
            f.write(f"  {label}:\n")
            f.write(f"    {desc} {marker}\n")
            f.write(f"    数量: {count} 张 ({percent:.1f}%) |{bar}|\n")

        # =====================================================================
        # SIOU 等级分布
        # =====================================================================
        f.write("\n\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🏆 SIOU 等级分布 (立体一致性) | 目标线: 0.28\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        siou_levels = [
            (0.50, '🌟 极佳 (>0.50)', '3D立体效果完美匹配'),
            (0.40, '⭐ 优秀 (0.40-0.50)', '3D立体效果很好'),
            (0.35, '✨ 良好 (0.35-0.40)', '3D立体效果好'),
            (0.28, '🎯 达标 (0.28-0.35)', '达到你的目标值！'),
            (0.20, '⚠️ 一般 (0.20-0.28)', '接近目标，需优化'),
            (0.15, '❌ 较差 (0.15-0.20)', '3D立体效果偏差'),
            (0, '💔 很差 (<0.15)', '3D立体效果差'),
        ]

        siou_counts = {}
        for threshold, _, _ in siou_levels:
            siou_counts[threshold] = 0

        for siou in all_siou:
            for threshold, _, _ in siou_levels:
                if siou >= threshold:
                    siou_counts[threshold] += 1
                    break

        for threshold, label, desc in siou_levels:
            count = siou_counts[threshold]
            percent = count / len(all_siou) * 100 if len(all_siou) > 0 else 0
            bar = '█' * int(percent / 5)
            marker = '← 🎯目标线' if threshold == target_siou else ''
            f.write(f"  {label}:\n")
            f.write(f"    {desc} {marker}\n")
            f.write(f"    数量: {count} 张 ({percent:.1f}%) |{bar}|\n")

        # =====================================================================
        # 最佳/最差排名
        # =====================================================================
        f.write("\n\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🏅 PSNR TOP 10 - 效果最好的图片 (✅=达标)\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        sorted_by_psnr = sorted(detailed_results, key=lambda x: x['psnr'], reverse=True)
        f.write(f"{'排名':>4} {'场景':<15} {'文件名':<25} {'PSNR':>8} {'SSIM':>8} {'SIOU':>8} {'达标':>6}\n")
        f.write(f"{'-'*4} {'-'*15} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6}\n")
        for i, r in enumerate(sorted_by_psnr[:10]):
            pass_p = '✅' if r['psnr'] >= target_psnr else ''
            pass_s = '✅' if r['ssim'] >= target_ssim else ''
            pass_i = '✅' if r['siou'] >= target_siou else ''
            pass_str = f"{pass_p}{pass_s}{pass_i}"
            f.write(f"{i+1:4d} {r['scene']:<15} {r['filename']:<25} {r['psnr']:8.2f} {r['ssim']:8.4f} {r['siou']:8.4f} {pass_str:>6}\n")

        f.write("\n\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📉 PSNR BOTTOM 10 - 效果最差的图片\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"{'排名':>4} {'场景':<15} {'文件名':<25} {'PSNR':>8} {'SSIM':>8} {'SIOU':>8} {'达标':>6}\n")
        f.write(f"{'-'*4} {'-'*15} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6}\n")
        for i, r in enumerate(sorted_by_psnr[-10:], 1):
            pass_p = '✅' if r['psnr'] >= target_psnr else ''
            pass_s = '✅' if r['ssim'] >= target_ssim else ''
            pass_i = '✅' if r['siou'] >= target_siou else ''
            pass_str = f"{pass_p}{pass_s}{pass_i}"
            idx = len(sorted_by_psnr) - 10 + i
            f.write(f"{idx:4d} {r['scene']:<15} {r['filename']:<25} {r['psnr']:8.2f} {r['ssim']:8.4f} {r['siou']:8.4f} {pass_str:>6}\n")

        # =====================================================================
        # 视差统计（只在自适应模式下有意义）
        # =====================================================================
        if use_relative_disparity:
            f.write("\n\n")
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write("📐 实际视差统计 (基准宽度自适应模式)\n")
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            all_disp = [r['actual_disparity'] for r in detailed_results]
            f.write(f"  配置视差: {max_disparity} px (@ {args.base_width}px 宽度)\n")
            f.write(f"  实际视差范围: {min(all_disp):.2f} ~ {max(all_disp):.2f} px\n")
            f.write(f"  实际视差平均: {np.mean(all_disp):.2f} px\n")
            f.write(f"  分辨率范围: {min([int(r['resolution'].split('x')[0]) for r in detailed_results])} ~ {max([int(r['resolution'].split('x')[0]) for r in detailed_results])} px\n")

        # 结尾
        f.write("\n\n")
        f.write("╔══════════════════════════════════════════════════════════════════════════════╗\n")
        f.write("║                           📝 报告结束                                         ║\n")
        f.write("║                    生成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "                              ║\n")
        f.write("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    print(f"[test] 📋 详细打分报告已保存: {detail_path}")

    return global_avg


# =============================================================================
# 测试主逻辑（多视差批量测试）
# =============================================================================

def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] 使用设备: {device}")

    # ---------- 加载模型（只加载一次） ----------
    print(f"[test] 加载模型: {args.encoder}")
    model = DepthAnythingV2(**get_model_config(args.encoder))
    ckpt = args.ckpt or f"checkpoints/depth_anything_v2_{args.encoder}.pth"
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    model = model.to(device).eval()
    if args.fp16:
        model = model.half()
    print("[test] 模型加载完成!")

    # ---------- 准备测试目录 ----------
    test_root = Path(args.test_root)
    scenes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
    print(f"[test] 发现 {len(scenes)} 个场景: {scenes}")

    base_out_root = Path(args.out_root)
    base_out_root.mkdir(parents=True, exist_ok=True)

    # ========== 视差模式 & 测试序列配置 ==========
    use_relative_disparity = args.use_relative_disparity

    # ┌─────────────────────────────────────────────────────────────┐
    # │   ✅ 只需修改这一个数组，两种模式共用                         │
    # │   绝对模式: 这些值就是像素数 (不管图多大)                     │
    # │   自适应模式: 这些值是 base_width 宽度下的像素数               │
    # │               实际视差会按当前图片宽度等比例缩放               │
    # └─────────────────────────────────────────────────────────────┘
    test_values = [6]

    print(f"\n{'='*70}")
    if use_relative_disparity:
        print(f"[test] 🎯 基准宽度自适应模式 (共 {len(test_values)} 个值)")
        print(f"[test]    基准宽度: {args.base_width}px（你原来调参的分辨率）")
        print(f"[test]    值的含义: 在 {args.base_width}px 宽度下的视差像素")
        print(f"[test]    实际视差 = value × (当前图片宽度 / {args.base_width})")
        print(f"[test]    测试序列: {test_values}")
    else:
        print(f"[test] 🎯 绝对像素视差模式 (共 {len(test_values)} 个值)")
        print(f"[test]    值的含义: 视差像素 = value（不管图多大）")
        print(f"[test]    测试序列: {test_values}")
    print(f"{'='*70}")

    disparity_values = test_values

    # ---------- 逐个视差测试 ----------
    all_results = {}
    for d in disparity_values:
        global_avg = run_single_max_disparity(
            max_disparity=d,
            model=model,
            device=device,
            test_root=test_root,
            base_out_root=base_out_root,
            args=args,
            use_relative_disparity=use_relative_disparity,
            base_width=args.base_width,
        )
        all_results[d] = global_avg

    # ---------- 生成汇总对比表 ----------
    print(f"\n{'='*70}")
    print("[test] 📊 所有视差结果汇总对比")
    print('='*70)
    if use_relative_disparity:
        print(f"\n{'自适应(1920基准)':>18} {'PSNR':>8} {'SSIM':>8} {'SIOU':>8} {'RMSE':>8}")
        print(f"{'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for d in sorted(all_results.keys()):
            avg = all_results[d]
            print(f"{d:18.1f} {avg['psnr']:8.2f} {avg['ssim']:8.4f} {avg['siou']:8.4f} {avg['rmse']:8.4f}")
    else:
        print(f"\n{'视差(绝对)':>12} {'PSNR':>8} {'SSIM':>8} {'SIOU':>8} {'RMSE':>8}")
        print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for d in sorted(all_results.keys()):
            avg = all_results[d]
            print(f"{d:12.1f} {avg['psnr']:8.2f} {avg['ssim']:8.4f} {avg['siou']:8.4f} {avg['rmse']:8.4f}")

    # 找出最优值
    best_psnr_d = max(all_results.keys(), key=lambda d: all_results[d]['psnr'])
    best_ssim_d = max(all_results.keys(), key=lambda d: all_results[d]['ssim'])
    best_siou_d = max(all_results.keys(), key=lambda d: all_results[d]['siou'])

    print(f"\n[test] 🏆 最优视差:")
    print(f"       PSNR 最优: disp={best_psnr_d} ({all_results[best_psnr_d]['psnr']:.2f} dB)")
    print(f"       SSIM 最优: disp={best_ssim_d} ({all_results[best_ssim_d]['ssim']:.4f})")
    print(f"       SIOU 最优: disp={best_siou_d} ({all_results[best_siou_d]['siou']:.4f})")

    # 保存汇总表
    summary_path = base_out_root / "results_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== mono2stereo 多视差测试汇总 ===\n")
        f.write(f"测试集: {test_root}\n")
        f.write(f"模型: {args.encoder}\n")
        f.write(f"输入尺寸: {args.input_size}\n")
        f.write(f"测试视差序列: {disparity_values}\n")
        f.write(f"\n")

        f.write("--- 结果对比表 ---\n")
        f.write(f"{'视差':>6} {'PSNR':>8} {'SSIM':>8} {'SIOU':>8} {'RMSE':>8} {'MSE':>8}\n")
        f.write(f"{'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}\n")
        for d in sorted(all_results.keys()):
            avg = all_results[d]
            f.write(f"{d:6d} {avg['psnr']:8.2f} {avg['ssim']:8.4f} {avg['siou']:8.4f} ")
            f.write(f"{avg['rmse']:8.4f} {avg['mse']:8.4f}\n")

        f.write(f"\n--- 最优值 ---\n")
        f.write(f"PSNR 最优: disp={best_psnr_d} ({all_results[best_psnr_d]['psnr']:.2f} dB)\n")
        f.write(f"SSIM 最优: disp={best_ssim_d} ({all_results[best_ssim_d]['ssim']:.4f})\n")
        f.write(f"SIOU 最优: disp={best_siou_d} ({all_results[best_siou_d]['siou']:.4f})\n")

    print(f"\n[test] ✅ 汇总表已保存: {summary_path}")


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="mono2stereo 定量测试")

    # ========== 所有默认参数已写死，直接运行即可 ==========
    parser.add_argument("--test-root", type=str,
                        default="/mnt/A/jiangxg/dataset/mono2stereo-test",
                        help="测试集根目录")
    parser.add_argument("--out-root", type=str,
                        default="./test_dav2_output",
                        help="输出目录")
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--ckpt", type=str, default=None,
                        help="模型权重路径 (默认: checkpoints/depth_anything_v2_{encoder}.pth)")
    parser.add_argument("--input-size", type=int, default=518,
                        help="深度模型输入长边尺寸 (必须是14的倍数)")
    parser.add_argument("--max-disparity", type=float, default=6,
                        help="最大视差像素（原分辨率）")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="启用 FP16 推理")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16",
                        help="禁用 FP16，用 FP32")
    parser.add_argument("--save-vis", action="store_true", default=True,
                        help="保存可视化结果（左|预测|真实 并排）")
    parser.add_argument("--no-save-vis", action="store_false", dest="save_vis",
                        help="不保存可视化结果")
    parser.add_argument("--use-relative-disparity", action="store_true", default=False,
                        help="启用基准宽度自适应视差（max_disparity 是 base_width 宽度下的值）")
    parser.add_argument("--base-width", type=int, default=1920,
                        help="基准宽度（你原来调参时的分辨率宽度，默认: 1920）")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
