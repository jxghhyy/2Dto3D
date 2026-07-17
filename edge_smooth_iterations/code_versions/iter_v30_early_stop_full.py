"""
迭代 v30: 迭代提前终止 + 完整阶段视频输出 ✨

核心改进：
1. ✅ 迭代提前终止：每轮检测填充进度，无进展立即停止（不需要硬编码 32 次）
2. ✅ 完整 9 阶段视频输出（与 v23_playable 完全一致）
3. ✅ 保留兜底填补逻辑（v29_fixed 修复）
"""
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.')
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# 全局核缓存
_KERNEL_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="v30: 迭代提前终止 + 完整阶段输出")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")

    parser.add_argument("--sharpen-threshold", type=float, default=3.0)
    parser.add_argument("--band-min-drop", type=float, default=3.5)
    parser.add_argument("--band-min-width", type=int, default=8)

    parser.add_argument("--edge-kernel-size", type=int, default=5, help="边缘小核")
    parser.add_argument("--non-edge-kernel-size", type=int, default=11, help="内部大核")
    parser.add_argument("--max-iter", type=int, default=64, help="最大迭代（有提前终止）")

    parser.add_argument("--left-smooth-width", type=int, default=6, help="左边界垂直平滑宽度")
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5, help="左边界平滑 sigma")
    parser.add_argument("--right-blend-width", type=int, default=3, help="右边缘羽化宽度")

    return parser.parse_args()


def _get_inpaint_kernels(kernel_size, device, dtype_str):
    key = (kernel_size, str(device), dtype_str)
    if key not in _KERNEL_CACHE:
        kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=getattr(torch, dtype_str))
        kernel3 = kernel1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[key] = (kernel1, kernel3)
    return _KERNEL_CACHE[key]


def detect_hole_edges(hole_mask):
    is_hole = hole_mask.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones((1, 1, 3, 3), device=hole_mask.device, dtype=is_hole.dtype)
    neighbor_count = F.conv2d(1 - is_hole, kernel, padding=1)
    edge_mask = hole_mask & (neighbor_count[0, 0] > 0.01)
    return edge_mask


def fill_edge_with_nearest_bg(img, hole, edge_mask, near, bg_threshold=0.3):
    result = img.clone()
    filled_mask = torch.zeros_like(hole)
    h, w = hole.shape
    device = hole.device

    bg_mask = ~hole & (near < bg_threshold)

    if torch.any(edge_mask):
        edge_extended = edge_mask.clone()
        for _ in range(2):
            edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=1) | torch.roll(edge_extended, shifts=-1, dims=1)
            edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=0) | torch.roll(edge_extended, shifts=-1, dims=0)

        to_fill = edge_extended & hole

        if torch.any(to_fill):
            img_nchw = img.permute(2, 0, 1).unsqueeze(0)
            bg_mask_nchw = bg_mask.unsqueeze(0).unsqueeze(0).float()

            pad = 2
            kernel1 = torch.ones((1, 1, 5, 5), device=device, dtype=img.dtype)
            kernel3 = kernel1.repeat(3, 1, 1, 1)

            weighted_img = img_nchw * bg_mask_nchw
            rgb_sum = F.conv2d(weighted_img, kernel3, padding=pad, groups=3)
            weight_sum = F.conv2d(bg_mask_nchw, kernel1, padding=pad)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            valid_fill = to_fill & (weight_sum[0, 0] > 0.5)
            if torch.any(valid_fill):
                result[valid_fill] = avg_hwc[valid_fill]
                filled_mask[valid_fill] = True

    return result, filled_mask


def project_disocclusion_bands_optimized(disparity, min_drop=3.5, min_band_width=8):
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    device = disparity.device
    right_cleanup = 10

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


@torch.no_grad()
def fast_inpaint_v30(img, hole, near, bg_threshold=0.3,
                      edge_kernel_size=5, non_edge_kernel_size=11, max_iter=64):
    """
    v30 改进版：带迭代提前终止的空洞填补
    ✅ 提前终止：每轮填充像素为 0 时立即停止
    ✅ 保留兜底填补逻辑
    """
    original_img = img.clone()
    fill_status = "normal"

    if not torch.any(hole):
        return img, 0, "无空洞"

    device = img.device
    h, w = hole.shape

    # ========== 步骤 1：检测边缘 + 直接填边缘 ==========
    edge_mask = detect_hole_edges(hole)
    img, edge_filled = fill_edge_with_nearest_bg(img, hole, edge_mask, near, bg_threshold)
    hole[edge_filled] = False

    if not torch.any(hole):
        return img, 0, "边缘直接填补完成"

    # ========== 步骤 2：大小核迭代填补（带提前终止） ==========
    edge_pad = edge_kernel_size // 2
    non_edge_pad = non_edge_kernel_size // 2
    edge_kernel1, edge_kernel3 = _get_inpaint_kernels(edge_kernel_size, device, str(img.dtype).split('.')[-1])
    non_edge_kernel1, non_edge_kernel3 = _get_inpaint_kernels(non_edge_kernel_size, device, str(img.dtype).split('.')[-1])

    is_bg = (near < bg_threshold).to(img.dtype)
    bg_w = is_bg.unsqueeze(0).unsqueeze(0)

    actual_iter = 0
    no_progress_count = 0
    prev_hole_count = hole.sum().item()

    for _ in range(max_iter):
        actual_iter += 1
        if not torch.any(hole):
            fill_status = "正常收敛"
            break

        filled_this_iter = 0
        known = (~hole).float().unsqueeze(0).unsqueeze(0)
        w = known * bg_w

        current_edge_mask = detect_hole_edges(hole) & hole

        if torch.any(current_edge_mask):
            edge_count = F.conv2d(w, edge_kernel1, padding=edge_pad)
            edge_fillable = current_edge_mask & (edge_count[0, 0] > 0.01)

            if torch.any(edge_fillable):
                img_nchw = img.permute(2, 0, 1).unsqueeze(0)
                edge_rgb_sum = F.conv2d(img_nchw * w, edge_kernel3, padding=edge_pad, groups=3)
                edge_avg = edge_rgb_sum / edge_count.clamp_min(1e-6)
                edge_avg_hwc = edge_avg[0].permute(1, 2, 0)

                img[edge_fillable] = edge_avg_hwc[edge_fillable]
                hole[edge_fillable] = False
                filled_this_iter += edge_fillable.sum().item()

        if torch.any(hole):
            non_edge_mask = hole & (~current_edge_mask)
            if torch.any(non_edge_mask):
                non_edge_count = F.conv2d(w, non_edge_kernel1, padding=non_edge_pad)
                non_edge_fillable = non_edge_mask & (non_edge_count[0, 0] > 0.01)

                if torch.any(non_edge_fillable):
                    img_nchw = img.permute(2, 0, 1).unsqueeze(0)
                    non_edge_rgb_sum = F.conv2d(img_nchw * w, non_edge_kernel3, padding=non_edge_pad, groups=3)
                    non_edge_avg = non_edge_rgb_sum / non_edge_count.clamp_min(1e-6)
                    non_edge_avg_hwc = non_edge_avg[0].permute(1, 2, 0)

                    img[non_edge_fillable] = non_edge_avg_hwc[non_edge_fillable]
                    hole[non_edge_fillable] = False
                    filled_this_iter += non_edge_fillable.sum().item()

        # ========== ✅ 提前终止检测 ==========
        current_hole_count = hole.sum().item()
        if filled_this_iter == 0 or current_hole_count >= prev_hole_count:
            no_progress_count += 1
            if no_progress_count >= 2:  # 连续 2 轮无进展，进入兜底
                fill_status = "提前终止 → 进入兜底"
                break
        else:
            no_progress_count = 0
        prev_hole_count = current_hole_count

    # ========== 步骤 3：兜底填补（与 mono2stereo 一致） ==========
    fallback_iter = 0
    if torch.any(hole):
        fill_status = "使用兜底填补"
        for _ in range(max_iter // 2):
            fallback_iter += 1
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

    # ========== 步骤 4：最差情况 - 全图平均填补 ==========
    if torch.any(hole):
        fill_status = f"使用全图平均填补 ({hole.sum().item()}像素)"
        known_pixels = original_img[~hole]
        fallback = known_pixels.mean(dim=0) if known_pixels.numel() > 0 else torch.zeros(3, device=device, dtype=img.dtype)
        img[hole] = fallback

    return img, actual_iter + fallback_iter, fill_status


@torch.no_grad()
def edge_post_process(image, hole_mask, smooth_width=6, smooth_sigma=1.5, blend_width=3):
    """左边界垂直平滑 + 右边缘羽化"""
    h, w = hole_mask.shape
    device = image.device

    result = image.clone()

    # 左边界垂直平滑
    if smooth_width > 0:
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        k = 5
        pad = k // 2
        sigma = smooth_sigma
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask = torch.zeros_like(hole_mask)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + smooth_width) & hole_mask
        smooth_mask_3d = smooth_region.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    # 右边缘羽化
    if blend_width > 0:
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) == 0:
                continue

            right_edge_x = cols.max()
            if right_edge_x + 1 >= w:
                continue

            blend_start_x = max(cols.min(), right_edge_x - blend_width + 1)
            blend_x_range = torch.arange(blend_start_x, right_edge_x + 1, device=device)

            if len(blend_x_range) == 0:
                continue

            dist_from_edge = right_edge_x - blend_x_range
            alpha = (dist_from_edge.float() / blend_width).clamp(0.0, 1.0).view(-1, 1)

            inpainted_colors = result[y, blend_x_range]
            bg_color = result[y, right_edge_x + 1]
            blended = alpha * inpainted_colors + (1 - alpha) * bg_color
            result[y, blend_x_range] = blended

    return result


def disparity_to_color(disp_np, max_val=None):
    """将视差图转换为彩色可视化（伪彩色）"""
    if max_val is None:
        max_val = disp_np.max()
    norm = (disp_np / max_val * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return color


def tensor_to_np_bgr(tensor_rgb):
    """GPU tensor [0,1] RGB → CPU uint8 BGR"""
    return cv2.cvtColor((tensor_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def mask_overlay(img_bgr, mask, color=(0, 0, 255), alpha=0.5):
    """在图像上叠加红色 mask"""
    overlay = img_bgr.copy()
    overlay[mask] = color
    return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v30 迭代提前终止 + 完整阶段输出 ✨")
    print(f"{'='*70}")
    print(f"  核心改进: 迭代提前终止（不需要硬编码上限）")
    print(f"  输出: 9 个阶段视频（与 v23_playable 一致）")
    print(f"{'='*70}\n")

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

    # 创建 9 个视频写入器（与 v23_playable 完全一致）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writers = {
        'b01_raw': cv2.VideoWriter(str(outdir / 'b01_raw_disparity.mp4'), fourcc, fps, (width, height)),
        'b02_sharp': cv2.VideoWriter(str(outdir / 'b02_sharpened_disparity.mp4'), fourcc, fps, (width, height)),
        'b03_excluded': cv2.VideoWriter(str(outdir / 'b03_excluded_pixels_yellow.mp4'), fourcc, fps, (width, height)),
        'b04_warped': cv2.VideoWriter(str(outdir / 'b04_warped_after_exclude.mp4'), fourcc, fps, (width, height)),
        'b05_hole_red': cv2.VideoWriter(str(outdir / 'b05_hole_after_exclude_red.mp4'), fourcc, fps, (width, height)),
        'c06_band_green': cv2.VideoWriter(str(outdir / 'c06_disocclusion_band_green.mp4'), fourcc, fps, (width, height)),
        'c07_hole_band': cv2.VideoWriter(str(outdir / 'c07_hole_with_band_red.mp4'), fourcc, fps, (width, height)),
        'c08_inpaint': cv2.VideoWriter(str(outdir / 'c08_inpaint_final.mp4'), fourcc, fps, (width, height)),
        'c09_stereo_sbs': cv2.VideoWriter(str(outdir / 'c09_final_stereo_sbs.mp4'), fourcc, fps, (width * 2, height)),
    }

    print(f"📹 已创建 9 个视频写入器\n")

    import time

    inpaint_times = []
    iter_counts = []
    status_counts = {}

    for frame_idx in tqdm(range(total_frames), ncols=80):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h_orig, w_orig = frame_bgr.shape[:2]
        left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 深度推理
        img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
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

        # 上采样到原分辨率
        near_score = F.interpolate(
            depth_norm[None, None, :, :],
            size=(h_orig, w_orig),
            mode="bilinear", align_corners=False
        )[0, 0]

        disparity = near_score * 24.0

        # ========== b01: 原始视差图 ==========
        disp_color_b01 = disparity_to_color(disparity.cpu().numpy())
        writers['b01_raw'].write(disp_color_b01)

        # 锐化视差
        disparity_sharp, unreliable = b.sharpen_disparity_edges(
            disparity, kernel_size=15, threshold=args.sharpen_threshold,
            iterations=1, reject_margin=0.10
        )

        # ========== b02: 锐化后视差图 ==========
        disp_color_b02 = disparity_to_color(disparity_sharp.cpu().numpy())
        writers['b02_sharp'].write(disp_color_b02)

        # DIBR 扭曲
        left_rgb = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        # ========== b03: 排除像素（黄色标记） ==========
        excluded_vis = tensor_to_np_bgr(right_warped)
        excluded_vis = mask_overlay(excluded_vis, unreliable.cpu().numpy(), color=(0, 255, 255), alpha=0.6)
        writers['b03_excluded'].write(excluded_vis)

        # ========== b04: 扭曲后图像 ==========
        writers['b04_warped'].write(tensor_to_np_bgr(right_warped))

        # ========== b05: 空洞（红色标记） ==========
        hole_vis_b05 = tensor_to_np_bgr(right_warped)
        hole_vis_b05 = mask_overlay(hole_vis_b05, hole.cpu().numpy(), color=(0, 0, 255), alpha=0.7)
        writers['b05_hole_red'].write(hole_vis_b05)

        # 反遮挡带
        disocclusion_band = project_disocclusion_bands_optimized(
            disparity_sharp,
            min_drop=args.band_min_drop,
            min_band_width=args.band_min_width
        )
        hole_with_band = hole | disocclusion_band

        # ========== c06: 反遮挡带（绿色标记） ==========
        band_vis = tensor_to_np_bgr(right_warped)
        band_vis = mask_overlay(band_vis, disocclusion_band.cpu().numpy(), color=(0, 255, 0), alpha=0.6)
        writers['c06_band_green'].write(band_vis)

        # ========== c07: 含反遮挡带的空洞（红色） ==========
        hole_band_vis = right_warped.clone()
        hole_band_vis[hole_with_band] = 0.0
        writers['c07_hole_band'].write(tensor_to_np_bgr(hole_band_vis))

        right_with_band = right_warped.clone()
        right_with_band[hole_with_band] = 0.0

        # ========== 计时开始：空洞填补 ==========
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # v30 带提前终止的大小核填补
        img_inpainted, actual_iter, status = fast_inpaint_v30(
            right_with_band, hole_with_band.clone(), near_score,
            bg_threshold=0.3,
            edge_kernel_size=args.edge_kernel_size,
            non_edge_kernel_size=args.non_edge_kernel_size,
            max_iter=args.max_iter
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inpaint_times.append((t1 - t0) * 1000)
        iter_counts.append(actual_iter)
        status_counts[status] = status_counts.get(status, 0) + 1

        # 边缘后处理
        final_right = edge_post_process(
            img_inpainted, hole_with_band,
            smooth_width=args.left_smooth_width,
            smooth_sigma=args.left_smooth_sigma,
            blend_width=args.right_blend_width
        )

        # ========== c08: 最终填补结果 ==========
        writers['c08_inpaint'].write(tensor_to_np_bgr(final_right))

        # ========== c09: 最终立体并排 ==========
        left_bgr = frame_bgr  # 原始左图 BGR
        final_right_bgr = tensor_to_np_bgr(final_right)
        stereo_sbs = np.hstack([left_bgr, final_right_bgr])
        writers['c09_stereo_sbs'].write(stereo_sbs)

    # 清理
    cap.release()
    for w in writers.values():
        w.release()

    # 统计信息
    avg_time = np.mean(inpaint_times)
    p95_time = np.percentile(inpaint_times, 95)
    max_time = np.max(inpaint_times)
    avg_iter = np.mean(iter_counts)

    print(f"\n{'='*70}")
    print(f"✅ v30 处理完成！结果保存在: {outdir}")
    print(f"\n⏱️  填补时间统计 (v30 带提前终止):")
    print(f"  平均: {avg_time:.2f} ms")
    print(f"  P95:  {p95_time:.2f} ms")
    print(f"  最大: {max_time:.2f} ms")
    print(f"\n📊 迭代次数统计:")
    print(f"  平均: {avg_iter:.1f} 次/帧")
    print(f"\n📈 填补状态统计:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count} 帧 ({count/total_frames*100:.1f}%)")
    print(f"\n🎬 生成的 9 个阶段视频:")
    stage_names = [
        ("b01_raw_disparity.mp4", "原始视差图（伪彩色）"),
        ("b02_sharpened_disparity.mp4", "锐化后视差图"),
        ("b03_excluded_pixels_yellow.mp4", "排除像素（黄色标记）"),
        ("b04_warped_after_exclude.mp4", "DIBR 扭曲后图像"),
        ("b05_hole_after_exclude_red.mp4", "空洞（红色标记）"),
        ("c06_disocclusion_band_green.mp4", "反遮挡带（绿色标记）"),
        ("c07_hole_with_band_red.mp4", "含反遮挡带的空洞"),
        ("c08_inpaint_final.mp4", "空洞填补最终结果"),
        ("c09_final_stereo_sbs.mp4", "最终立体并排视频"),
    ]
    for fname, desc in stage_names:
        fpath = outdir / fname
        size_mb = fpath.stat().st_size / 1024 / 1024 if fpath.exists() else 0
        print(f"  {fname:40s} ({size_mb:.1f} MB) - {desc}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
