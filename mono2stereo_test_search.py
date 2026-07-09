"""
mono2stereo 每张图片最佳视差搜索脚本
功能：对每张图片测试多个视差值，找到 SIOU 最高的那个
输出：每张图片的最佳视差配置文件（供后续批量测评使用）
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
from skimage.metrics import structural_similarity as ssim

from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
from utils_depth.metrics import eval_stereo


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
    # ⚠️  除零保护：当两张图都没有边缘时，避免 0/0 = nan
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

    # ⚠️  NaN 保护：确保所有指标不会出现 nan
    for key in result:
        val = result[key]
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            if key == 'siou':
                result[key] = 0.0
            elif key == 'ssim':
                result[key] = 0.0
            elif key == 'psnr':
                result[key] = 0.0

    return result


# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════╗
# ║   🎯 在这里填写你要搜索的视差值序列（绝对像素，或者基准宽度下的值）  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# =============================================================================
# 你可以自由修改这个列表！
# 例1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# 例2: [0.5, 1, 2, 3, 4, 5, 6, 8, 10]
# 例3: list(np.arange(0.2, 5.0, 0.2))  # 0.2~5.0 步进 0.2
# =============================================================================
SEARCH_DISPARITIES = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,7,8,9,10,12,16,20,24,28,32,36,40]


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
    N = h * w
    device = left_rgb.device

    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    x_tgt = torch.round(xs.float() - disparity).long()

    valid = (x_tgt >= 0) & (x_tgt < w)
    src_lin = (ys * w + xs).reshape(-1)
    tgt_lin = (ys * w + x_tgt).reshape(-1)
    valid_flat = valid.reshape(-1)
    near_flat = disparity.reshape(-1)

    src_lin = src_lin[valid_flat]
    tgt_lin = tgt_lin[valid_flat]
    near_flat = near_flat[valid_flat]

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


@torch.no_grad()
def process_single_image_with_disparity(
    left_path: str,
    gt_right_path: str,
    model: DepthAnythingV2,
    device: torch.device,
    input_size: int = 518,
    disparity_value: float = 2.0,
    use_relative_disparity: bool = False,
    base_width: int = 1920,
    fp16: bool = True,
    use_buggy_metrics: bool = False,
) -> Dict:
    """
    处理单张左图 + 单个视差值，返回三个指标
    优化：深度图只推理一次，缓存起来！
    """
    left_bgr = cv2.imread(left_path)
    h_orig, w_orig = left_bgr.shape[:2]
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

    # ---------- 深度预处理 + 推理（只做一次！）----------
    # 🎯 只做14的倍数对齐，不做分辨率压缩！尽可能用原图尺寸进行深度估计
    # 只要是14的倍数，Depth-Anything 就能正常工作，不限制最大尺寸
    depth_h = max(14, ((h_orig + 13) // 14) * 14)  # 向上取最近的14的倍数
    depth_w = max(14, ((w_orig + 13) // 14) * 14)

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

    # ---------- 归一化（每帧独立，和之前保持一致）----------
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

    # ---------- 计算当前视差 ----------
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

    # ---------- 读取真实右图 + 计算指标 ----------
    gt_right_bgr = cv2.imread(gt_right_path)
    gt_right = cv2.cvtColor(gt_right_bgr, cv2.COLOR_BGR2RGB)
    if pred_right.shape != gt_right.shape:
        gt_right = cv2.resize(gt_right, (pred_right.shape[1], pred_right.shape[0]))

    # ⚠️  选择使用 buggy 或正确的 metrics
    if use_buggy_metrics:
        # uint8 溢出版本（复现论文结果）
        metrics = eval_stereo_buggy(pred_right, gt_right, left_rgb)
    else:
        # 修正版本（float32，无溢出）
        metrics = eval_stereo(pred_right, gt_right, left_rgb)

    # inf / nan 处理
    for key in metrics:
        val = metrics[key]
        if isinstance(val, float):
            if np.isinf(val):
                if key == 'psnr':
                    metrics[key] = 80.0
                else:
                    metrics[key] = 0.0
            elif np.isnan(val):
                metrics[key] = 0.0

    return {
        'disp_config': disparity_value,      # 你配置的视差值
        'disp_actual': actual_disparity,     # 实际视差（自适应模式下可能不同）
        'psnr': metrics['psnr'],
        'ssim': metrics['ssim'],
        'siou': metrics['siou'],
        'rmse': metrics['rmse'],
        'mse': metrics['mse'],
    }


def search_best_disparity_for_image(
    left_path: str,
    gt_right_path: str,
    model: DepthAnythingV2,
    device: torch.device,
    search_values: List[float],
    args,
) -> Dict:
    """
    对单张图片搜索所有视差值，返回 SIOU 最高的那个
    """
    all_results = []

    for disp in search_values:
        result = process_single_image_with_disparity(
            left_path, gt_right_path, model, device,
            input_size=args.input_size,
            disparity_value=disp,
            use_relative_disparity=args.use_relative_disparity,
            base_width=args.base_width,
            fp16=args.fp16,
            use_buggy_metrics=args.use_buggy_metrics,
        )
        all_results.append(result)

    # 🔧 智能排序：优先 SIOU，但如果所有 SIOU 都接近 0，退而求其次用 SSIM/PSNR
    # 计算所有有效 SIOU 的最大值
    valid_siou = [r['siou'] for r in all_results if r['siou'] > 0.001]
    max_siou = max(valid_siou) if valid_siou else 0

    if max_siou > 0.01:
        # 正常情况：有有效的 SIOU，按 SIOU 优先
        all_results.sort(key=lambda x: (x['siou'], x['ssim'], x['psnr']), reverse=True)
    else:
        # ⚠️  SIOU 全部失效（边缘检测失败）
        # 退而求其次：按 SSIM 优先，然后 PSNR
        all_results.sort(key=lambda x: (x['ssim'], x['psnr']), reverse=True)

    best = all_results[0]

    return {
        'best': best,
        'all_results': all_results,  # 所有视差的结果（用于画曲线）
    }


def main():
    parser = argparse.ArgumentParser(description="mono2stereo 每张图片最佳视差搜索")
    parser.add_argument("--test-root", type=str,
                        default="/mnt/A/jiangxg/dataset/mono2stereo-test",
                        help="测试集根目录")
    parser.add_argument("--out-root", type=str,
                        default="./test_output_search",
                        help="输出目录")
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--use-relative-disparity", action="store_true", default=True,
                        help="启用基准宽度自适应视差（推荐！）")
    parser.add_argument("--base-width", type=int, default=1920,
                        help="基准宽度（自适应模式下）")
    parser.add_argument("--save-top-curves", type=int, default=20,
                        help="保存前 N 张图的 SIOU-视差 曲线数据（用于分析）")
    parser.add_argument("--use-buggy-metrics", action="store_true",
                        help="⚠️  使用有 uint8 溢出 bug 的 metrics 计算（复现论文结果），会导致误差被低估")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[search] 使用设备: {device}")

    # ---------- 加载模型 ----------
    print(f"[search] 加载模型: Depth-Anything-V2-{args.encoder.upper()}")
    print(f"[search] 精度模式: {'FP16' if args.fp16 else 'FP32 (高精度)'}")
    model = DepthAnythingV2(**get_model_config(args.encoder))
    ckpt = f"submodules/depth/dav2/checkpoints/depth_anything_v2_{args.encoder}.pth"
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

    # ---------- 准备搜索 ----------
    test_root = Path(args.test_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    search_values = SEARCH_DISPARITIES
    print(f"\n{'='*70}")
    print(f"[search] 🎯 最佳视差搜索模式")
    if args.use_buggy_metrics:
        print(f"[search] ⚠️  METRICS: 使用有 uint8 溢出 BUG 的版本（复现论文结果）")
        print(f"[search]    ⚠️  注意: MSE/RMSE/PSNR 会因溢出被严重低估！")
    else:
        print(f"[search] ✓  METRICS: 使用修正后的版本（float32，无溢出）")
    if args.use_relative_disparity:
        print(f"[search]    模式: 基准宽度自适应 (base_width = {args.base_width})")
    else:
        print(f"[search]    模式: 绝对像素视差")
    print(f"[search]    深度分辨率: 原图尺寸 (仅对齐14的倍数，不压缩)")
    print(f"[search]    搜索视差值: {search_values}")
    print(f"[search]    择优依据: SIOU 优先 → SSIM → PSNR")
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
                all_image_pairs.append({
                    'scene': scene,
                    'left': str(left_path),
                    'right': str(right_path),
                    'filename': left_path.name,
                })

    print(f"[search] 共发现 {len(all_image_pairs)} 张图片待搜索\n")

    # =====================================================================
    # 逐张搜索最佳视差
    # =====================================================================
    best_disparities = []  # 存储所有图片的最佳视差结果
    fallback_count = 0  # 统计降级使用 SSIM 的案例数

    for i, pair in enumerate(all_image_pairs):
        scene = pair['scene']
        filename = pair['filename']

        print(f"[{i+1}/{len(all_image_pairs)}] {scene}/{filename} ... ", end='', flush=True)

        result = search_best_disparity_for_image(
            pair['left'], pair['right'], model, device, search_values, args
        )

        best = result['best']

        # 检查是否是降级判断
        best['is_fallback'] = best['siou'] < 0.01
        if best['is_fallback']:
            fallback_count += 1

        best_disparities.append({
            'scene': scene,
            'filename': filename,
            'left_path': pair['left'],
            'best_disp': best['disp_config'],
            'best_disp_actual': best['disp_actual'],
            'best_siou': best['siou'],
            'best_ssim': best['ssim'],
            'best_psnr': best['psnr'],
            'is_fallback': best['is_fallback'],
            'all_results': result['all_results'],
        })

        fallback_marker = " (⚠️  SSIM 降级)" if best.get('is_fallback', False) else ""
        print(f"最佳视差 = {best['disp_config']:.2f}, "
              f"SIOU = {best['siou']:.4f}, "
              f"SSIM = {best['ssim']:.4f}, "
              f"PSNR = {best['psnr']:.2f}{fallback_marker}")

    # =====================================================================
    # 输出 1: 最佳视差配置文件（CSV 格式，供你后续批量测评使用）
    # =====================================================================
    csv_path = out_root / "best_disparities.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("scene,filename,best_disp,best_disp_actual,siou,ssim,psnr\n")
        for r in best_disparities:
            f.write(f"{r['scene']},{r['filename']},{r['best_disp']:.4f},{r['best_disp_actual']:.4f},")
            f.write(f"{r['best_siou']:.6f},{r['best_ssim']:.6f},{r['best_psnr']:.2f}\n")

    print(f"\n[search] ✅ CSV 配置文件已保存: {csv_path}")

    # =====================================================================
    # 输出 2: 简化版配置文件（只有 路径 → 视差，方便读取）
    # =====================================================================
    simple_path = out_root / "disparity_config.txt"
    with open(simple_path, 'w', encoding='utf-8') as f:
        f.write("# 格式: 场景名/文件名 最佳视差值\n")
        f.write("# 用于后续批量测评时，每张图片使用自己的最优视差\n\n")
        for r in best_disparities:
            f.write(f"{r['scene']}/{r['filename']} {r['best_disp']:.4f}\n")

    print(f"[search] ✅ 精简配置文件已保存: {simple_path}")

    # =====================================================================
    # 输出 3: 详细统计分析报告
    # =====================================================================
    report_path = out_root / "search_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("╔══════════════════════════════════════════════════════════════════════════╗\n")
        f.write("║                 📊 mono2stereo 最佳视差搜索分析报告                        ║\n")
        f.write("╚══════════════════════════════════════════════════════════════════════════╝\n\n")

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📋 基本信息\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"搜索时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.use_buggy_metrics:
            f.write("⚠️  METRICS: 使用有 uint8 溢出 BUG 的版本（复现论文结果）\n")
            f.write("   ⚠️  注意: MSE/RMSE/PSNR 会因溢出被严重低估！\n")
        else:
            f.write("✓  METRICS: 使用修正后的版本（float32，无溢出）\n")
        f.write(f"搜索模式: {'基准宽度自适应' if args.use_relative_disparity else '绝对像素'}\n")
        if args.use_relative_disparity:
            f.write(f"基准宽度: {args.base_width}px\n")
        f.write(f"搜索视差值: {search_values}\n")
        f.write(f"搜索图片数: {len(best_disparities)}\n")
        f.write(f"择优依据: SIOU 优先 → SSIM → PSNR\n\n")

        # 最佳视差分布统计
        all_best_disp = [r['best_disp'] for r in best_disparities]
        all_best_siou = [r['best_siou'] for r in best_disparities]
        all_best_ssim = [r['best_ssim'] for r in best_disparities]
        all_best_psnr = [r['best_psnr'] for r in best_disparities]

        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📉 最佳视差分布统计\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"  最佳视差最小值: {min(all_best_disp):.4f}\n")
        f.write(f"  最佳视差最大值: {max(all_best_disp):.4f}\n")
        f.write(f"  最佳视差平均值: {np.mean(all_best_disp):.4f}\n")
        f.write(f"  最佳视差中位数: {np.median(all_best_disp):.4f}\n")
        f.write(f"  最佳视差标准差: {np.std(all_best_disp):.4f}\n\n")

        # 视差值出现频率
        f.write("  最佳视差值出现频率:\n")
        unique_disp, counts = np.unique(np.round(all_best_disp, 2), return_counts=True)
        for d, c in sorted(zip(unique_disp, counts), key=lambda x: -x[1]):
            pct = c / len(all_best_disp) * 100
            bar = '█' * int(pct / 3)
            f.write(f"    disp = {d:.2f}: {c:4d} 张 ({pct:5.1f}%) {bar}\n")

        f.write("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("🏆 采用最优视差后的全局指标（理想上限）\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write(f"  SIOU 平均值: {np.mean(all_best_siou):.4f} (目标: 0.28)\n")
        f.write(f"  SIOU 中位数: {np.median(all_best_siou):.4f}\n")
        f.write(f"  SIOU ≥ 0.28: {sum(1 for s in all_best_siou if s >= 0.28)} 张 "
                f"({sum(1 for s in all_best_siou if s >= 0.28)/len(all_best_siou)*100:.1f}%)\n")
        if fallback_count > 0:
            f.write(f"  ⚠️  SIOU 失效 (≈0): {fallback_count} 张 "
                    f"({fallback_count/len(all_best_siou)*100:.1f}%) - 已降级使用 SSIM 排序\n\n")
        else:
            f.write("\n")

        f.write(f"  SSIM 平均值: {np.mean(all_best_ssim):.4f} (目标: 0.75)\n")
        f.write(f"  SSIM 中位数: {np.median(all_best_ssim):.4f}\n")
        f.write(f"  SSIM ≥ 0.75: {sum(1 for s in all_best_ssim if s >= 0.75)} 张 "
                f"({sum(1 for s in all_best_ssim if s >= 0.75)/len(all_best_ssim)*100:.1f}%)\n\n")

        f.write(f"  PSNR 平均值: {np.mean(all_best_psnr):.2f} dB (目标: 32)\n")
        f.write(f"  PSNR 中位数: {np.median(all_best_psnr):.2f} dB\n")
        f.write(f"  PSNR ≥ 32: {sum(1 for p in all_best_psnr if p >= 32)} 张 "
                f"({sum(1 for p in all_best_psnr if p >= 32)/len(all_best_psnr)*100:.1f}%)\n\n")

        # 三目标同时达标
        triple_pass = sum(1 for r in best_disparities
                          if r['best_siou'] >= 0.28 and r['best_ssim'] >= 0.75 and r['best_psnr'] >= 32)
        f.write(f"  三目标同时达标: {triple_pass} 张 ({triple_pass/len(best_disparities)*100:.1f}%)\n")

        # 按场景统计
        f.write("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        f.write("📊 按场景统计最佳视差\n")
        f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        scene_stats = {}
        for r in best_disparities:
            s = r['scene']
            if s not in scene_stats:
                scene_stats[s] = {'disps': [], 'siou': [], 'ssim': [], 'psnr': [], 'count': 0}
            scene_stats[s]['disps'].append(r['best_disp'])
            scene_stats[s]['siou'].append(r['best_siou'])
            scene_stats[s]['ssim'].append(r['best_ssim'])
            scene_stats[s]['psnr'].append(r['best_psnr'])
            scene_stats[s]['count'] += 1

        f.write(f"{'场景':<15} {'图片数':>8} {'最佳视差(avg)':>16} {'SIOU(avg)':>12} {'SSIM(avg)':>12} {'PSNR(avg)':>12}\n")
        f.write(f"{'-'*15} {'-'*8} {'-'*16} {'-'*12} {'-'*12} {'-'*12}\n")
        for s in sorted(scene_stats.keys()):
            stats = scene_stats[s]
            f.write(f"{s:<15} {stats['count']:8d} {np.mean(stats['disps']):16.4f} ")
            f.write(f"{np.mean(stats['siou']):12.4f} {np.mean(stats['ssim']):12.4f} {np.mean(stats['psnr']):12.2f}\n")

        # TOP 20 曲线数据（用于后续分析）
        if args.save_top_curves > 0:
            f.write("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            f.write(f"📈 前 {args.save_top_curves} 张图片的 SIOU-视差 曲线数据\n")
            f.write("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

            sorted_by_siou = sorted(best_disparities, key=lambda x: x['best_siou'], reverse=True)
            for i, r in enumerate(sorted_by_siou[:args.save_top_curves]):
                f.write(f"[{i+1}] {r['scene']}/{r['filename']} (最佳 SIOU = {r['best_siou']:.4f})\n")
                f.write(f"  视差:  {[x['disp_config'] for x in r['all_results']]}\n")
                f.write(f"  SIOU:  {[round(x['siou'], 6) for x in r['all_results']]}\n")
                f.write(f"  SSIM:  {[round(x['ssim'], 6) for x in r['all_results']]}\n")
                f.write(f"  PSNR:  {[round(x['psnr'], 2) for x in r['all_results']]}\n\n")

    print(f"[search] ✅ 分析报告已保存: {report_path}")

    # =====================================================================
    # 终局提示
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"[search] 🏆 搜索完成！")
    print(f"{'='*70}")
    print(f"\n  📊 理想上限（每张图片用各自最优视差）:")
    print(f"    SIOU 平均: {np.mean(all_best_siou):.4f}  (目标: 0.28)")
    print(f"    SSIM 平均: {np.mean(all_best_ssim):.4f}  (目标: 0.75)")
    print(f"    PSNR 平均: {np.mean(all_best_psnr):.2f} dB (目标: 32)")

    # 降级统计
    fallback_count = sum(1 for r in best_disparities if r.get('is_fallback', False))
    if fallback_count > 0:
        print(f"\n  ⚠️  SIOU 失效案例: {fallback_count}/{len(best_disparities)} 张")
        print(f"      这些图片使用了 SSIM 作为降级排序依据")

    print(f"\n  💡 后续步骤:")
    print(f"    1. 查看 {simple_path} 或 {csv_path}")
    print(f"    2. 修改测评脚本，读取这个视差配置文件")
    print(f"    3. 跑一遍，看看实际测评结果能提升多少")
    print(f"    4. 如果提升明显，下一步就是预测每张图的最佳视差了！")
    print()


if __name__ == "__main__":
    main()
