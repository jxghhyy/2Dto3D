"""
mono2stereo 视差配置文件测评脚本
功能：读取 disparity_config.txt，每张图片用自己的最佳视差运行
方法：纯 DIBR + 均值扩散补洞（无 SD 补救）
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


# =============================================================================
# ⚠️  100% 精确复现原始 metrics_ago.py 的 uint8 溢出 bug
# 仅用于复现论文报告的数值，真实评估请使用修正后的 metrics
# =============================================================================
def detect_edges_buggy(image, low, high):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges


def edge_overlap_buggy(edge1, edge2):
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    # 除零保护
    if union == 0:
        return 0.0
    return intersection / union


def compute_siou_buggy(pred, target, left):
    """100%精确复现 metrics_ago.py 的 SIoU 计算"""
    left_edges = detect_edges_buggy(left, 100, 200)
    pred_edges = detect_edges_buggy(pred, 100, 200)
    right_edges = detect_edges_buggy(target, 100, 200)

    # ⚠️  Python 内置 abs()，在 uint8 上直接计算（会溢出）
    diff_gl = abs(pred - left)
    diff_rl = abs(target - left)

    diff_gl = cv2.cvtColor(diff_gl, cv2.COLOR_BGR2GRAY)
    diff_rl = cv2.cvtColor(diff_rl, cv2.COLOR_BGR2GRAY)
    diff_gl_ = np.zeros(diff_rl.shape)
    diff_rl_ = np.zeros(diff_rl.shape)
    diff_gl_[diff_gl > 5] = 1
    diff_rl_[diff_rl > 5] = 1

    edge_overlap_gr = edge_overlap_buggy(pred_edges, right_edges)
    diff_overlap_grl = edge_overlap_buggy(diff_gl_, diff_rl_)

    return 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl


def eval_stereo_buggy(pred, target, left):
    """⚠️  100%精确复现原始 metrics_ago.py 的 uint8 溢出 bug

    完全复刻论文发表时的计算逻辑，uint8 下的减法、平方、绝对值都会溢出。
    仅用于复现论文报告的数值，真实评估请使用修正后的 metrics。
    """
    from skimage.metrics import structural_similarity as ssim

    max_pixel = 255.0
    assert pred.shape == target.shape

    # ⚠️  BUG #1: uint8 直接相减，负数值会溢出
    # 例如：pred=10, target=200 → diff=-190 → uint8 溢出变成 66
    diff = pred - target

    # ⚠️  BUG #2: uint8 下直接平方，大数值会再次溢出
    # 例如：diff=66 → 66²=4356 → 4356 mod 256 = 84 ❌
    mse_err = np.mean(diff ** 2)

    rmse = np.sqrt(mse_err)

    # ⚠️  BUG #3: uint8 下取绝对值
    absolute_errors = np.abs(diff)
    mae = np.mean(absolute_errors)

    # 原始代码没有 rmse==0 的判断
    psnr = 20 * np.log10(max_pixel / rmse)

    # SSIM 不受影响，skimage 内部会转 float
    ssim_value, _ = ssim(pred, target, full=True, multichannel=True,
                         win_size=7, channel_axis=2)

    # SIoU 计算（完全复刻原始版本）
    siou_value = compute_siou_buggy(pred, target, left)

    # 原始代码用 .item()
    result = {
        'rmse': rmse.item() if hasattr(rmse, 'item') else float(rmse),
        'mse': mse_err.item() if hasattr(mse_err, 'item') else float(mse_err),
        'mae': mae.item() if hasattr(mae, 'item') else float(mae),
        'siou': siou_value.item() if hasattr(siou_value, 'item') else float(siou_value),
        'psnr': psnr.item() if hasattr(psnr, 'item') else float(psnr),
        'ssim': ssim_value.item() if hasattr(ssim_value, 'item') else float(ssim_value),
    }

    # NaN 保护
    for key in result:
        val = result[key]
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            if key == 'psnr':
                result[key] = 80.0 if np.isinf(val) else 0.0
            else:
                result[key] = 0.0

    return result


# =============================================================================
# 修正版 metrics（无溢出）
# =============================================================================
def eval_stereo_correct(pred, target, left):
    """修正版 metrics（float32 计算，无溢出）"""
    from skimage.metrics import structural_similarity as ssim

    max_pixel = 255.0
    assert pred.shape == target.shape

    pred_f = pred.astype(np.float32)
    target_f = target.astype(np.float32)

    diff = pred_f - target_f
    mse_err = np.mean(diff ** 2)
    rmse = np.sqrt(mse_err)

    psnr = 20 * np.log10(max_pixel / rmse) if rmse > 0 else 80.0

    ssim_value, _ = ssim(pred, target, full=True, multichannel=True,
                         win_size=7, channel_axis=2)

    # 正确的 SIOU 计算（用 float32）
    left_edges = cv2.Canny(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), 100, 200)
    pred_edges = cv2.Canny(cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY), 100, 200)
    right_edges = cv2.Canny(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), 100, 200)

    intersection = np.logical_and(pred_edges, right_edges).sum()
    union = np.logical_or(pred_edges, right_edges).sum()
    edge_overlap_gr = intersection / union if union > 0 else 0.0

    # 正确的差异计算
    diff_gl = np.abs(pred.astype(np.float32) - left.astype(np.float32))
    diff_rl = np.abs(target.astype(np.float32) - left.astype(np.float32))
    diff_gl = cv2.cvtColor(diff_gl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    diff_rl = cv2.cvtColor(diff_rl.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    diff_gl_ = (diff_gl > 5).astype(np.float32)
    diff_rl_ = (diff_rl > 5).astype(np.float32)

    intersection = np.logical_and(diff_gl_, diff_rl_).sum()
    union = np.logical_or(diff_gl_, diff_rl_).sum()
    diff_overlap_grl = intersection / union if union > 0 else 0.0

    siou_value = 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl

    return {
        'rmse': float(rmse),
        'mse': float(mse_err),
        'mae': float(np.mean(np.abs(diff))),
        'siou': float(siou_value),
        'psnr': float(psnr) if not np.isinf(psnr) else 80.0,
        'ssim': float(ssim_value),
    }


def get_model_config(encoder: str) -> Dict:
    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    return configs[encoder]


@torch.no_grad()
def forward_warp_right_gpu(
    left_rgb: torch.Tensor, disparity: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w, _ = left_rgb.shape
    device = left_rgb.device

    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    x_tgt = torch.round(xs.float() - disparity).long()

    valid = (x_tgt >= 0) & (x_tgt < w)
    src_lin = (ys * w + xs).reshape(-1)
    tgt_lin = (ys * w + x_tgt).reshape(-1)
    valid_flat = valid.reshape(-1)

    src_lin = src_lin[valid_flat]
    tgt_lin = tgt_lin[valid_flat]

    NEAR_BITS = 20
    near_q = torch.ones_like(src_lin) * ((1 << NEAR_BITS) - 1)
    encoded = (near_q << 32) | src_lin.long()

    max_encoded = torch.full((h * w,), -1, device=device, dtype=torch.int64)
    max_encoded.scatter_reduce_(0, tgt_lin, encoded, reduce="amax", include_self=True)
    selected = encoded == max_encoded[tgt_lin]

    src_sel = src_lin[selected]
    tgt_sel = tgt_lin[selected]

    left_flat = left_rgb.reshape(-1, 3)
    right_flat = torch.zeros_like(left_flat)
    right_flat[tgt_sel] = left_flat[src_sel]

    hole = torch.ones((h * w,), device=device, dtype=torch.bool)
    hole[tgt_sel] = False

    return right_flat.reshape(h, w, 3), hole.reshape(h, w)


@torch.no_grad()
def dilate_hole_right(hole: torch.Tensor, dilate_px: int = 1) -> torch.Tensor:
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

    if torch.any(hole):
        fallback = img[~hole_mask].mean(dim=0)
        img[hole] = fallback

    return img


def load_disparity_config(config_path: str) -> Dict[str, float]:
    """
    读取视差配置文件
    返回: {"场景名/文件名": 视差值}
    """
    config = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]  # 格式: scene/filename
                disp = float(parts[1])
                config[key] = disp
    print(f"[config] 已加载 {len(config)} 个视差配置")
    return config


@torch.no_grad()
def process_single_image(
    left_path: str,
    disparity_value: float,
    model: DepthAnythingV2,
    device: torch.device,
    input_size: int = 518,
    use_relative_disparity: bool = True,
    base_width: int = 1920,
    fp16: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    DIBR + 均值扩散补洞
    返回: (pred_right, left_rgb, actual_disparity)
    """
    left_bgr = cv2.imread(left_path)
    h_orig, w_orig = left_bgr.shape[:2]
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

    # ---------- 深度预处理 + 推理（518 长边模式）----------
    long_edge = input_size
    scale = long_edge / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = long_edge
    else:
        depth_h = long_edge

    img = cv2.resize(left_rgb, (depth_w, depth_h), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    if fp16:
        img_tensor = img_tensor.half()

    depth = model(img_tensor)[0].float()

    # ---------- 归一化 ----------
    flat = depth.reshape(-1)
    sample_size = min(16384, flat.numel())
    idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=device))
    q_low, q_high = q_vals[0], q_vals[1]
    denom = (q_high - q_low).clamp_min(1e-6)
    depth_norm = ((depth - q_low) / denom).clamp(0.0, 1.0)
    near = depth_norm

    # ---------- 上采样到原分辨率 ----------
    near_orig = F.interpolate(
        near[None, None, :, :], size=(h_orig, w_orig),
        mode="bilinear", align_corners=False,
    )[0, 0]

    # ---------- 计算视差 ----------
    if use_relative_disparity:
        scale_factor = w_orig / base_width
        actual_disparity = disparity_value * scale_factor
    else:
        actual_disparity = disparity_value

    disparity = near_orig * actual_disparity

    # ---------- DIBR + 补洞 ----------
    left_gpu = torch.from_numpy(left_rgb).to(device=device, dtype=torch.float32) / 255.0
    right_warped, hole = forward_warp_right_gpu(left_gpu, disparity)
    hole_dilated = dilate_hole_right(hole, dilate_px=1)
    right_inpainted = fast_inpaint_gpu(right_warped, hole_dilated, kernel_size=11, max_iter=64)
    pred_right = (right_inpainted.clamp(0, 1) * 255.0).byte().cpu().numpy()

    return pred_right, left_rgb, actual_disparity


def main():
    parser = argparse.ArgumentParser(description="mono2stereo 视差配置文件测评（纯 DIBR）")
    parser.add_argument("--test-root", type=str,
                        default="/mnt/A/jiangxg/dataset/mono2stereo-test",
                        help="测试集根目录")
    parser.add_argument("--out-root", type=str,
                        default="./test_output_with_config",
                        help="输出目录")
    parser.add_argument("--config", type=str,
                        default="./test_output_search/disparity_config.txt",
                        help="视差配置文件路径")
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--use-relative-disparity", action="store_true", default=True,
                        help="启用基准宽度自适应视差（必须和 search 时一致！）")
    parser.add_argument("--base-width", type=int, default=1920,
                        help="基准宽度（必须和 search 时一致！）")
    parser.add_argument("--default-disp", type=float, default=2.0,
                        help="配置文件中找不到时的默认视差值")
    parser.add_argument("--use-buggy-metrics", action="store_true",
                        help="⚠️  使用有 uint8 溢出 bug 的 metrics 计算（复现论文结果）")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] 使用设备: {device}")

    # ---------- 加载视差配置 ----------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] 找不到配置文件: {config_path}")
        print(f"        请先运行 mono2stereo_test_search.py 生成配置文件")
        return
    disp_config = load_disparity_config(str(config_path))

    # ---------- 加载 Depth-Anything 模型 ----------
    print(f"[test] 加载模型: Depth-Anything-V2-{args.encoder.upper()}")
    print(f"[test] 精度模式: {'FP16' if args.fp16 else 'FP32 (高精度)'}")
    model = DepthAnythingV2(**get_model_config(args.encoder))
    ckpt = f"/mnt/A/jiangxg/work/2Dto3D/submodules/Mono2Stereo/depth/checkpoints/depth_anything_v2_{args.encoder}.pth"
    if not os.path.exists(ckpt):
        ckpt = f"checkpoints/depth_anything_v2_{args.encoder}.pth"
    if not os.path.exists(ckpt):
        print(f"[warning] 找不到权重: {ckpt}，尝试自动下载...")
        from torch.hub import load_state_dict_from_url
        url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{args.encoder.upper()}/resolve/main/depth_anything_v2_{args.encoder}.pth"
        state_dict = load_state_dict_from_url(url, map_location=device)
    else:
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    if args.fp16:
        model = model.half()

    # ---------- 准备输出 ----------
    test_root = Path(args.test_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[test] 🎯 视差配置文件测评模式（纯 DIBR）")
    if args.use_buggy_metrics:
        print(f"[test] ⚠️  METRICS: 使用有 uint8 溢出 BUG 的版本（复现论文结果）")
    else:
        print(f"[test] ✓  METRICS: 使用修正后的版本（float32，无溢出）")
    if args.use_relative_disparity:
        print(f"[test]    模式: 基准宽度自适应 (base_width = {args.base_width})")
    else:
        print(f"[test]    模式: 绝对像素视差")
    print(f"[test]    深度模型: Depth-Anything-V2-{args.encoder} (518长边)")
    print(f"[test]    配置文件: {config_path}")
    print(f"[test]    配置数量: {len(disp_config)} 张图片")
    print(f"[test]    默认视差: {args.default_disp}")
    print(f"[test]    输出目录: {out_root}/")
    print(f"{'='*70}\n")

    # ---------- 收集所有图片 ----------
    all_image_pairs = []
    scenes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])

    for scene in scenes:
        left_dir = test_root / scene / "left"
        right_dir = test_root / scene / "right"
        if not left_dir.exists() or not right_dir.exists():
            continue
        left_files = sorted([f for f in left_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        for left_path in left_files:
            right_path = right_dir / left_path.name
            if right_path.exists():
                key = f"{scene}/{left_path.name}"
                disp = disp_config.get(key, args.default_disp)
                all_image_pairs.append({
                    'scene': scene,
                    'left': str(left_path),
                    'right': str(right_path),
                    'filename': left_path.name,
                    'key': key,
                    'disp_config': disp,
                    'has_config': key in disp_config,
                })

    has_config_count = sum(1 for p in all_image_pairs if p['has_config'])
    print(f"[test] 共 {len(all_image_pairs)} 张图片，其中 {has_config_count} 张有配置，"
          f"{len(all_image_pairs) - has_config_count} 张使用默认值\n")

    # =====================================================================
    # 逐张测评
    # =====================================================================
    all_results = []
    missing_config = []

    for i, pair in enumerate(all_image_pairs):
        scene = pair['scene']
        filename = pair['filename']
        disp_config_val = pair['disp_config']
        has_config = pair['has_config']

        if not has_config:
            missing_config.append(pair['key'])

        print(f"[{i+1}/{len(all_image_pairs)}] {scene}/{filename} "
              f"disp={disp_config_val:.2f} {'(默认)' if not has_config else ''} ... ",
              end='', flush=True)

        try:
            pred_right, left_rgb, actual_disp = process_single_image(
                pair['left'], disp_config_val, model, device,
                input_size=args.input_size,
                use_relative_disparity=args.use_relative_disparity,
                base_width=args.base_width,
                fp16=args.fp16,
            )

            # 计算指标
            gt_right_bgr = cv2.imread(pair['right'])
            gt_right = cv2.cvtColor(gt_right_bgr, cv2.COLOR_BGR2RGB)
            if pred_right.shape != gt_right.shape:
                gt_right = cv2.resize(gt_right, (pred_right.shape[1], pred_right.shape[0]))

            if args.use_buggy_metrics:
                metrics = eval_stereo_buggy(pred_right, gt_right, left_rgb)
            else:
                metrics = eval_stereo_correct(pred_right, gt_right, left_rgb)

            all_results.append({
                'scene': scene,
                'filename': filename,
                'resolution': f"{pred_right.shape[1]}x{pred_right.shape[0]}",
                'disp_config': disp_config_val,
                'disp_actual': actual_disp,
                'has_config': has_config,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'siou': metrics['siou'],
                'rmse': metrics['rmse'],
                'mse': metrics['mse'],
            })

            # 保存图像处理结果
            img_out_dir = out_root / "results" / scene
            img_out_dir.mkdir(parents=True, exist_ok=True)
            # 1. 左图
            cv2.imwrite(str(img_out_dir / f"left_{filename}"), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
            # 2. 生成的右图
            cv2.imwrite(str(img_out_dir / f"pred_right_{filename}"), cv2.cvtColor(pred_right, cv2.COLOR_RGB2BGR))
            # 3. 并排对比
            h, w = left_rgb.shape[:2]
            gt_right_bgr = cv2.imread(pair['right'])
            gt_right = cv2.cvtColor(gt_right_bgr, cv2.COLOR_BGR2RGB)
            if gt_right.shape[:2] != pred_right.shape[:2]:
                gt_right = cv2.resize(gt_right, (w, h), interpolation=cv2.INTER_CUBIC)
            vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
            vis[:, :w] = left_rgb
            vis[:, w:2*w] = pred_right
            vis[:, 2*w:] = gt_right
            cv2.imwrite(str(img_out_dir / f"compare_{filename}"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            print(f"SIOU={metrics['siou']:.4f}, SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # =====================================================================
    # 输出 1: CSV 完整结果
    # =====================================================================
    csv_path = out_root / "results_full.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("scene,filename,resolution,disp_config,disp_actual,has_config,siou,ssim,psnr,rmse,mse\n")
        for r in all_results:
            f.write(f"{r['scene']},{r['filename']},{r['resolution']},{r['disp_config']:.4f},{r['disp_actual']:.4f},")
            f.write(f"{1 if r['has_config'] else 0},{r['siou']:.6f},{r['ssim']:.6f},{r['psnr']:.2f},")
            f.write(f"{r['rmse']:.6f},{r['mse']:.6f}\n")

    print(f"\n[test] ✅ 完整结果已保存: {csv_path}")

    # =====================================================================
    # 输出 2: 超级详细报告
    # =====================================================================
    report_path = out_root / "detailed_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("╔══════════════════════════════════════════════════════════════════════════╗\n")
        f.write("║                   📊 视差配置文件测评详细报告                              ║\n")
        f.write("╚══════════════════════════════════════════════════════════════════════════╝\n\n")

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📋 基本信息\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"测评时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.use_buggy_metrics:
            f.write("⚠️  METRICS: 使用有 uint8 溢出 BUG 的版本（复现论文结果）\n")
        else:
            f.write("✓  METRICS: 使用修正后的版本（float32，无溢出）\n")
        f.write(f"视差模式: {'基准宽度自适应' if args.use_relative_disparity else '绝对像素'}\n")
        if args.use_relative_disparity:
            f.write(f"基准宽度: {args.base_width}px\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"模型: Depth-Anything-V2-{args.encoder}\n")
        f.write(f"总图片数: {len(all_results)}\n")
        f.write(f"有配置图片数: {has_config_count} ({has_config_count/len(all_results)*100:.1f}%)\n")
        f.write(f"默认视差值: {args.default_disp}\n\n")

        if missing_config:
            f.write("⚠️  缺少配置的图片（使用默认值）:\n")
            for key in missing_config[:20]:
                f.write(f"  - {key}\n")
            if len(missing_config) > 20:
                f.write(f"  ... 还有 {len(missing_config) - 20} 张\n")
            f.write("\n")

        # 按场景分组显示
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🖼️  逐张详细结果（按场景分组）\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        current_scene = None
        scene_metrics = {}

        for r in all_results:
            scene = r['scene']
            if scene != current_scene:
                if current_scene is not None:
                    # 上一个场景的统计
                    avg = scene_metrics[current_scene]
                    f.write(f"\n  📈 场景「{current_scene}」平均:\n")
                    f.write(f"     SIOU 平均: {avg['siou']:.4f}\n")
                    f.write(f"     SSIM 平均: {avg['ssim']:.4f}\n")
                    f.write(f"     PSNR 平均: {avg['psnr']:.2f} dB\n")
                    f.write(f"     平均视差: {avg['disp']:.4f}\n")
                    f.write(f"{'-'*78}\n\n")

                current_scene = scene
                f.write(f"\n{'='*78}\n")
                f.write(f"场景: {scene}\n")
                f.write(f"{'='*78}\n")
                f.write(f"{'序号':>4} {'文件名':<25} {'配置视差':>10} {'实际视差':>10} {'SIOU':>8} {'SSIM':>8} {'PSNR':>8}\n")
                f.write(f"{'-'*4} {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}\n")
                scene_metrics[scene] = {'siou': [], 'ssim': [], 'psnr': [], 'disp': []}

            scene_metrics[scene]['siou'].append(r['siou'])
            scene_metrics[scene]['ssim'].append(r['ssim'])
            scene_metrics[scene]['psnr'].append(r['psnr'])
            scene_metrics[scene]['disp'].append(r['disp_config'])

            idx = len(scene_metrics[scene]['siou'])
            marker = '*' if not r['has_config'] else ' '
            f.write(f"{idx:4d}{marker} {r['filename']:<25} {r['disp_config']:10.4f} {r['disp_actual']:10.4f} ")
            f.write(f"{r['siou']:8.4f} {r['ssim']:8.4f} {r['psnr']:8.2f}\n")

        # 最后一个场景
        if current_scene is not None:
            avg = scene_metrics[current_scene]
            f.write(f"\n  📈 场景「{current_scene}」平均:\n")
            f.write(f"     SIOU 平均: {np.mean(avg['siou']):.4f}\n")
            f.write(f"     SSIM 平均: {np.mean(avg['ssim']):.4f}\n")
            f.write(f"     PSNR 平均: {np.mean(avg['psnr']):.2f} dB\n")
            f.write(f"     平均视差: {np.mean(avg['disp']):.4f}\n")

        # =====================================================================
        # 全局统计
        # =====================================================================
        all_siou = [r['siou'] for r in all_results]
        all_ssim = [r['ssim'] for r in all_results]
        all_psnr = [r['psnr'] for r in all_results]
        all_disp = [r['disp_config'] for r in all_results]

        f.write("\n\n")
        f.write("╔══════════════════════════════════════════════════════════════════════════╗\n")
        f.write("║                           📊 全局统计汇总                                 ║\n")
        f.write("╚══════════════════════════════════════════════════════════════════════════╝\n\n")

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📈 各场景对比汇总\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"{'场景名':<15} {'图片数':>8} {'平均视差':>12} {'SIOU(avg)':>12} {'SSIM(avg)':>12} {'PSNR(avg)':>12}\n")
        f.write(f"{'-'*15} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n")

        for s in sorted(scene_metrics.keys()):
            m = scene_metrics[s]
            f.write(f"{s:<15} {len(m['siou']):8d} {np.mean(m['disp']):12.4f} ")
            f.write(f"{np.mean(m['siou']):12.4f} {np.mean(m['ssim']):12.4f} {np.mean(m['psnr']):12.2f}\n")

        f.write("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📉 指标分布统计 (全部图片)\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        f.write(f"\n  SIOU (目标: 0.28):\n")
        f.write(f"    最小值: {np.min(all_siou):.4f}\n")
        f.write(f"    最大值: {np.max(all_siou):.4f}\n")
        f.write(f"    平均值: {np.mean(all_siou):.4f}\n")
        f.write(f"    中位数: {np.median(all_siou):.4f}\n")
        f.write(f"    标准差: {np.std(all_siou):.4f}\n")
        f.write(f"    ≥ 0.28: {sum(1 for s in all_siou if s >= 0.28)} 张 ({sum(1 for s in all_siou if s >= 0.28)/len(all_siou)*100:.1f}%)\n")

        f.write(f"\n  SSIM (目标: 0.75):\n")
        f.write(f"    最小值: {np.min(all_ssim):.4f}\n")
        f.write(f"    最大值: {np.max(all_ssim):.4f}\n")
        f.write(f"    平均值: {np.mean(all_ssim):.4f}\n")
        f.write(f"    中位数: {np.median(all_ssim):.4f}\n")
        f.write(f"    标准差: {np.std(all_ssim):.4f}\n")
        f.write(f"    ≥ 0.75: {sum(1 for s in all_ssim if s >= 0.75)} 张 ({sum(1 for s in all_ssim if s >= 0.75)/len(all_ssim)*100:.1f}%)\n")

        f.write(f"\n  PSNR (目标: 32 dB):\n")
        f.write(f"    最小值: {np.min(all_psnr):.2f}\n")
        f.write(f"    最大值: {np.max(all_psnr):.2f}\n")
        f.write(f"    平均值: {np.mean(all_psnr):.2f}\n")
        f.write(f"    中位数: {np.median(all_psnr):.2f}\n")
        f.write(f"    标准差: {np.std(all_psnr):.2f}\n")
        f.write(f"    ≥ 32: {sum(1 for p in all_psnr if p >= 32)} 张 ({sum(1 for p in all_psnr if p >= 32)/len(all_psnr)*100:.1f}%)\n")

        # 三目标同时达标
        triple_pass = sum(1 for r in all_results
                          if r['siou'] >= 0.28 and r['ssim'] >= 0.75 and r['psnr'] >= 32)
        f.write(f"\n  三目标同时达标: {triple_pass} 张 ({triple_pass/len(all_results)*100:.1f}%)\n")

        # =====================================================================
        # SIOU 等级分布
        # =====================================================================
        f.write("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🏆 SIOU 等级分布\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        siou_levels = [
            (0.40, '🌟 极佳 (>0.40)'),
            (0.35, '✨ 优秀 (0.35-0.40)'),
            (0.28, '🎯 达标 (0.28-0.35)'),
            (0.20, '⚠️  一般 (0.20-0.28)'),
            (0.15, '❌ 较差 (0.15-0.20)'),
            (0, '💔 很差 (<0.15)'),
        ]

        for threshold, label in siou_levels:
            if threshold == 0:
                count = sum(1 for s in all_siou if s < 0.15)
            else:
                next_t = next((t for t, _ in siou_levels if t < threshold), 0)
                if next_t == 0:
                    count = sum(1 for s in all_siou if s >= threshold)
                else:
                    count = sum(1 for s in all_siou if next_t <= s < threshold)
            percent = count / len(all_siou) * 100 if len(all_siou) > 0 else 0
            bar = '█' * int(percent / 3)
            f.write(f"  {label}: {count:5d} 张 ({percent:5.1f}%) {bar}\n")

        # =====================================================================
        # TOP/BOTTOM 榜单
        # =====================================================================
        f.write("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🏅 SIOU TOP 20 - 效果最好的图片\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        sorted_by_siou = sorted(all_results, key=lambda x: x['siou'], reverse=True)
        f.write(f"{'排名':>4} {'场景':<15} {'文件名':<25} {'视差':>8} {'SIOU':>8} {'SSIM':>8} {'PSNR':>8}\n")
        f.write(f"{'-'*4} {'-'*15} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}\n")
        for i, r in enumerate(sorted_by_siou[:20]):
            f.write(f"{i+1:4d} {r['scene']:<15} {r['filename']:<25} {r['disp_config']:8.2f} ")
            f.write(f"{r['siou']:8.4f} {r['ssim']:8.4f} {r['psnr']:8.2f}\n")

        f.write("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📉 SIOU BOTTOM 20 - 效果最差的图片\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"{'排名':>4} {'场景':<15} {'文件名':<25} {'视差':>8} {'SIOU':>8} {'SSIM':>8} {'PSNR':>8}\n")
        f.write(f"{'-'*4} {'-'*15} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}\n")
        for i, r in enumerate(sorted_by_siou[-20:], 1):
            idx = len(sorted_by_siou) - 20 + i
            f.write(f"{idx:4d} {r['scene']:<15} {r['filename']:<25} {r['disp_config']:8.2f} ")
            f.write(f"{r['siou']:8.4f} {r['ssim']:8.4f} {r['psnr']:8.2f}\n")

    print(f"[test] ✅ 详细报告已保存: {report_path}")

    # =====================================================================
    # 终局提示
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"[test] 🏆 测评完成！")
    print(f"{'='*70}")
    print(f"\n  📊 实际测评结果:")
    print(f"    SIOU 平均: {np.mean(all_siou):.4f}  (目标: 0.28)")
    print(f"    SSIM 平均: {np.mean(all_ssim):.4f}  (目标: 0.75)")
    print(f"    PSNR 平均: {np.mean(all_psnr):.2f} dB (目标: 32)")
    print(f"\n  💡 建议:")
    print(f"    1. 对比 search_analysis_report.txt 里的「理想上限」")
    print(f"    2. 看看差距有多大？如果差距很小，说明搜索结果很准！")
    print(f"    3. 如果差距很大，检查是不是 search 和 test 的参数不一致？")
    print()


if __name__ == "__main__":
    main()
