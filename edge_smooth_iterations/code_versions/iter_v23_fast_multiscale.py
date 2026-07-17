"""
迭代 v23_fast: 大小核卷积 - 速度优化版 ✨

对比 v23 的改进：
1. 每轮只填边缘能填的，不做全区域无效计算
2. 填不动立刻 break（不硬跑 max_iter）
3. uniform kernel 代替高斯（更快，效果足够）
4. 核缓存复用（不每帧新建）
5. 更少的 permute 操作
6. 左边界延迟处理但不改逻辑
"""
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import subprocess
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, '.')
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="v23_fast: 大小核卷积填补（速度优化版）")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--generate-side-by-side", action="store_true", help="生成并排对比视频")

    # B版本锐化参数
    parser.add_argument("--sharpen-kernel-size", type=int, default=15, help="深度锐化水平窗口宽度")
    parser.add_argument("--sharpen-threshold", type=float, default=3.0, help="触发锐化所需的局部视差跨度")
    parser.add_argument("--sharpen-reject-margin", type=float, default=0.10, help="过渡区域排除比例")

    # 反遮挡带参数
    parser.add_argument("--band-min-drop", type=float, default=3.5, help="生成反遮挡带所需的视差下降量")
    parser.add_argument("--band-right-cleanup", type=int, default=10, help="反遮挡带右端清除宽度")
    parser.add_argument("--band-min-width", type=int, default=8, help="反遮挡带最小有效宽度")

    # 大小核卷积参数
    parser.add_argument("--edge-kernel-size", type=int, default=5, help="边缘小核尺寸")
    parser.add_argument("--non-edge-kernel-size", type=int, default=11, help="内部大核尺寸")
    parser.add_argument("--max-iter", type=int, default=64, help="最大迭代次数")
    parser.add_argument("--left-margin-delay", type=int, default=8, help="左边界延迟处理宽度（像素）")

    # 边缘平滑 + 羽化参数
    parser.add_argument("--left-smooth-width", type=int, default=6, help="左边界垂直平滑宽度")
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5, help="左边界平滑 sigma")
    parser.add_argument("--right-blend-width", type=int, default=3, help="右边缘羽化宽度")

    return parser.parse_args()


# ==================== 核缓存 ====================
_KERNEL_CACHE = {}


def get_inpaint_kernels(kernel_size, device, dtype):
    """缓存 inpaint kernel，避免每帧重建"""
    key = (kernel_size, str(device), str(dtype))
    if key not in _KERNEL_CACHE:
        kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
        kernel3 = kernel1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[key] = (kernel1, kernel3)
    return _KERNEL_CACHE[key]


def detect_hole_edges(hole_mask):
    """检测空洞边缘：洞 + 至少一个邻居不是洞"""
    is_hole = hole_mask.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones((1, 1, 3, 3), device=hole_mask.device, dtype=is_hole.dtype)
    neighbor_count = F.conv2d(1 - is_hole, kernel, padding=1)
    edge_mask = hole_mask & (neighbor_count[0, 0] > 0.01)
    return edge_mask


# ==================== 核心：快速大小核填补（参考原版 fast_inpaint_gpu） ====================
def inpaint_multiscale_fast(image, hole_mask, near_score, bg_threshold=0.3,
                             edge_kernel_size=5, non_edge_kernel_size=11, max_iter=64,
                             left_margin_delay=8):
    """
    大小核卷积填补（速度优化版）

    优化点：
    1. 每轮只填边缘能填的像素
    2. 填不动立刻 break（不硬跑满 max_iter）
    3. uniform kernel（不是高斯，更快且效果足够）
    4. 核缓存复用
    5. 左边界延迟处理：最后才填，防止前景颜色扩散
    """
    h, w = hole_mask.shape
    device = image.device

    # 背景种子掩码
    is_bg = (near_score < bg_threshold).to(image.dtype).unsqueeze(0).unsqueeze(0)

    # 初始化
    img = image.clone()
    hole = hole_mask.clone()
    if not torch.any(hole):
        return img

    # 获取核（缓存）
    edge_pad = edge_kernel_size // 2
    non_edge_pad = non_edge_kernel_size // 2
    edge_kernel1, edge_kernel3 = get_inpaint_kernels(edge_kernel_size, device, img.dtype)
    non_edge_kernel1, non_edge_kernel3 = get_inpaint_kernels(non_edge_kernel_size, device, img.dtype)

    # ========== 步骤1：找到左边界延迟区域 ==========
    edge_x = torch.full((h,), w, device=device, dtype=torch.long)
    for y in range(h):
        cols = torch.where(hole_mask[y])[0]
        if len(cols) > 0:
            edge_x[y] = cols.min()

    x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
    edge_x_2d = edge_x.view(h, 1).expand(h, w)
    left_margin = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + left_margin_delay)
    left_margin = left_margin & hole_mask  # 只在初始空洞内的左边缘延迟

    # ========== 步骤2：先填非左边界区域 ==========
    hole_active = hole & (~left_margin)

    for _ in range(max_iter):
        if not torch.any(hole_active):
            break

        current_edge = detect_hole_edges(hole_active) & hole_active
        if not torch.any(current_edge):
            break

        known = (~hole_active).float().unsqueeze(0).unsqueeze(0)
        w = known * is_bg

        filled_this_iter = torch.zeros_like(hole_active)

        # --- 边缘用小核 ---
        edge_count = F.conv2d(w, edge_kernel1, padding=edge_pad)
        edge_fillable = current_edge & (edge_count[0, 0] > 0.01)

        if torch.any(edge_fillable):
            img_nchw = img.permute(2, 0, 1).unsqueeze(0)
            edge_rgb_sum = F.conv2d(img_nchw * w, edge_kernel3, padding=edge_pad, groups=3)
            edge_avg = edge_rgb_sum / edge_count.clamp_min(1e-6)
            edge_avg_hwc = edge_avg[0].permute(1, 2, 0)

            img[edge_fillable] = edge_avg_hwc[edge_fillable]
            hole_active[edge_fillable] = False
            filled_this_iter |= edge_fillable

        # --- 非边缘用大核（如果还有剩的） ---
        if torch.any(hole_active):
            non_edge_mask = hole_active & (~current_edge)
            if torch.any(non_edge_mask):
                non_edge_count = F.conv2d(w, non_edge_kernel1, padding=non_edge_pad)
                non_edge_fillable = non_edge_mask & (non_edge_count[0, 0] > 0.01)

                if torch.any(non_edge_fillable):
                    img_nchw = img.permute(2, 0, 1).unsqueeze(0)
                    non_edge_rgb_sum = F.conv2d(img_nchw * w, non_edge_kernel3, padding=non_edge_pad, groups=3)
                    non_edge_avg = non_edge_rgb_sum / non_edge_count.clamp_min(1e-6)
                    non_edge_avg_hwc = non_edge_avg[0].permute(1, 2, 0)

                    img[non_edge_fillable] = non_edge_avg_hwc[non_edge_fillable]
                    hole_active[non_edge_fillable] = False
                    filled_this_iter |= non_edge_fillable

        if not torch.any(filled_this_iter):
            break

    # ========== 步骤3：最后填左边界区域 ==========
    # 此时内部已经填好，用小核从右向左"扫"左边缘
    hole_left = hole & left_margin  # 只填初始左边界内的像素
    if torch.any(hole_left):
        # 更新 hole 状态（把之前填好的区域当作已知）
        hole = hole_left

        for _ in range(max_iter // 2):
            if not torch.any(hole):
                break

            current_edge = detect_hole_edges(hole) & hole
            if not torch.any(current_edge):
                break

            known = (~hole).float().unsqueeze(0).unsqueeze(0)

            edge_count = F.conv2d(known, edge_kernel1, padding=edge_pad)
            edge_fillable = current_edge & (edge_count[0, 0] > 0.01)

            if torch.any(edge_fillable):
                img_nchw = img.permute(2, 0, 1).unsqueeze(0)
                edge_rgb_sum = F.conv2d(img_nchw * known, edge_kernel3, padding=edge_pad, groups=3)
                edge_avg = edge_rgb_sum / edge_count.clamp_min(1e-6)
                edge_avg_hwc = edge_avg[0].permute(1, 2, 0)

                img[edge_fillable] = edge_avg_hwc[edge_fillable]
                hole[edge_fillable] = False

    # ========== 步骤4：兜底填补（如果还有剩下的洞） ==========
    if torch.any(hole):
        for _ in range(max_iter // 2):
            if not torch.any(hole):
                break

            fallback_known = (~hole).float().unsqueeze(0).unsqueeze(0)
            count = F.conv2d(fallback_known, non_edge_kernel1, padding=non_edge_pad)
            fillable = hole & (count[0, 0] > 0.01)

            if not torch.any(fillable):
                break

            img_nchw = img.permute(2, 0, 1).unsqueeze(0)
            rgb_sum = F.conv2d(img_nchw * fallback_known, non_edge_kernel3, padding=non_edge_pad, groups=3)
            avg = rgb_sum / count.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            img[fillable] = avg_hwc[fillable]
            hole[fillable] = False

    # 最差情况：还有洞，用全图平均
    if torch.any(hole):
        known_pixels = image[~hole_mask]
        fallback = known_pixels.mean(dim=0) if known_pixels.numel() > 0 else torch.zeros(3, device=device, dtype=img.dtype)
        img[hole] = fallback

    return img


# ==================== 辅助函数 ====================

def project_disocclusion_bands_optimized(
    disparity, min_drop=3.5, right_cleanup=10, min_band_width=8
):
    """优化的反遮挡带生成（参数可配置）"""
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    device = disparity.device

    x_left = torch.arange(w - 1, device=device).view(1, w - 1)
    d_left = disparity[:, :-1]
    d_right = disparity[:, 1:]
    is_drop = (d_left - d_right) >= min_drop

    foreground_target = x_left.to(disparity.dtype) - d_left
    background_target = (x_left + 1).to(disparity.dtype) - d_right
    start = torch.floor(foreground_target).long() + 1
    end = torch.floor(background_target).long() + right_cleanup
    start = start.clamp(0, w - 1)
    end = end.clamp(0, w - 1)
    valid = is_drop & (end >= start)

    difference = torch.zeros((h, w + 1), device=device, dtype=torch.int32)
    rows = torch.arange(h, device=device).view(h, 1).expand(h, w - 1)
    flat = difference.reshape(-1)
    start_index = (rows * (w + 1) + start)[valid]
    stop_index = (rows * (w + 1) + end + 1)[valid]
    flat.scatter_add_(0, start_index, torch.ones_like(start_index, dtype=flat.dtype))
    flat.scatter_add_(0, stop_index, -torch.ones_like(stop_index, dtype=flat.dtype))

    band = torch.cumsum(difference[:, :w], dim=1) > 0

    # 按宽度过滤小带
    band_np = band.cpu().numpy()
    for y in range(h):
        row = band_np[y]
        if row.sum() == 0:
            continue
        changes = np.diff(np.concatenate(([0], row.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < min_band_width:
                band_np[y, s:e] = False

    return torch.from_numpy(band_np).to(device)


def process_frame(frame_bgr, model, device, args):
    """处理单帧"""
    h_orig, w_orig = frame_bgr.shape[:2]

    # 深度推理
    img = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)
    img_resized = F.interpolate(img, size=(294, 518), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    model_input = (img_resized - mean_t) / std_t

    with torch.no_grad():
        depth_raw = model(model_input)[0].float()

    # 归一化
    flat = depth_raw.reshape(-1)
    idx = torch.randint(0, flat.numel(), (16384,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=flat.device))
    low, high = q_vals[0], q_vals[1]
    depth_norm = ((depth_raw - low) / (high - low)).clamp(0.0, 1.0)

    # 上采样
    near_score = F.interpolate(
        depth_norm[None, None, :, :],
        size=(h_orig, w_orig),
        mode="bilinear", align_corners=False
    )[0, 0]

    disparity = near_score * 24.0

    # B版本：锐化 + 排除过渡像素
    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity,
        kernel_size=args.sharpen_kernel_size,
        threshold=args.sharpen_threshold,
        iterations=1,
        reject_margin=args.sharpen_reject_margin
    )

    left_rgb = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    # 反遮挡带
    disocclusion_band = project_disocclusion_bands_optimized(
        disparity_sharp,
        min_drop=args.band_min_drop,
        right_cleanup=args.band_right_cleanup,
        min_band_width=args.band_min_width
    )
    hole_with_band = hole | disocclusion_band

    # ========== 原版 B 填补（用于对比） ==========
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    right_with_band_b = right_warped.clone()
    right_with_band_b[hole | disocclusion_band] = 0.0

    final_right_b = b.strict_background_inpaint_gpu_b(
        right_with_band_b, hole | disocclusion_band, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== v23_fast 大小核卷积填补 ==========
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    final_right_optimized = inpaint_multiscale_fast(
        right_with_band, hole_with_band, near_score,
        bg_threshold=0.3,
        edge_kernel_size=args.edge_kernel_size,
        non_edge_kernel_size=args.non_edge_kernel_size,
        max_iter=args.max_iter,
        left_margin_delay=args.left_margin_delay
    )

    # 左边界垂直平滑
    if args.left_smooth_width > 0:
        h, w = hole_with_band.shape
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_with_band[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        smooth_mask = torch.zeros_like(hole_with_band)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + args.left_smooth_width)
        smooth_mask = smooth_region & hole_with_band

        k = 5
        pad = k // 2
        sigma = args.left_smooth_sigma
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = final_right_optimized.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask_3d = smooth_mask.unsqueeze(-1)
        final_right_optimized = torch.where(smooth_mask_3d, img_smoothed, final_right_optimized)

    # 右边缘羽化融合
    if args.right_blend_width > 0:
        h, w = hole_with_band.shape
        for y in range(h):
            cols = torch.where(hole_with_band[y])[0]
            if len(cols) == 0:
                continue

            right_edge_x = cols.max()
            if right_edge_x + 1 >= w:
                continue

            blend_start_x = max(cols.min(), right_edge_x - args.right_blend_width + 1)
            blend_x_range = torch.arange(blend_start_x, right_edge_x + 1, device=device)

            if len(blend_x_range) == 0:
                continue

            dist_from_edge = right_edge_x - blend_x_range
            alpha = (dist_from_edge.float() / args.right_blend_width).clamp(0.0, 1.0).view(-1, 1)

            inpainted_colors = final_right_optimized[y, blend_x_range]
            bg_color = final_right_optimized[y, right_edge_x + 1]
            blended = alpha * inpainted_colors + (1 - alpha) * bg_color
            final_right_optimized[y, blend_x_range] = blended

    return final_right_b.cpu().numpy(), final_right_optimized.cpu().numpy()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v23_fast 大小核卷积（速度优化版）✨")
    print(f"{'='*60}")
    print(f"【核心参数】")
    print(f"  锐化阈值: {args.sharpen_threshold}, 反遮挡带阈值: {args.band_min_drop}")
    print(f"  大小核: {args.edge_kernel_size}x{args.edge_kernel_size} + {args.non_edge_kernel_size}x{args.non_edge_kernel_size}")
    print(f"  左边界延迟宽度: {args.left_margin_delay} 像素")
    print(f"  平滑宽度: {args.left_smooth_width}, 羽化宽度: {args.right_blend_width}")
    print(f"{'='*60}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # 打开视频
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {total_frames} 帧, {fps} fps, {width}x{height}\n")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_opt = cv2.VideoWriter(str(outdir / 'v23_fast_optimized_only.mp4'), fourcc, fps, (width, height))

    if args.generate_side_by_side:
        out_compare = cv2.VideoWriter(str(outdir / 'v23_fast_side_by_side_compare.mp4'), fourcc, fps, (width * 2, height))

    for frame_idx in tqdm(range(total_frames), ncols=80):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        result_b, result_opt = process_frame(frame_bgr, model, device, args)

        # 转换为 BGR
        opt_u8 = (result_opt * 255).astype(np.uint8)
        opt_bgr = cv2.cvtColor(opt_u8, cv2.COLOR_RGB2BGR)
        out_opt.write(opt_bgr)

        if args.generate_side_by_side:
            b_u8 = (result_b * 255).astype(np.uint8)
            b_bgr = cv2.cvtColor(b_u8, cv2.COLOR_RGB2BGR)
            side_by_side = np.hstack([b_bgr, opt_bgr])
            out_compare.write(side_by_side)

    cap.release()
    out_opt.release()
    if args.generate_side_by_side:
        out_compare.release()

    print(f"\n{'='*60}")
    print(f"✅ v23_fast 处理完成！结果保存在: {outdir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
