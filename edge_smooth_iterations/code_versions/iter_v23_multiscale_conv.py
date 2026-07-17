"""
迭代 v23: 大小核卷积填补版本 ✨

核心改进:
1. 大小核卷积：小核(5x5)处理边缘，大核(11x11)处理内部
2. 左边界延迟卷积：先卷内部，最后卷左边缘（防止前景信息扩散）
3. 完整调试输出：b01-b05, c06-c09 全阶段视频

注意：不使用 B 版本的镜像采样方法，纯卷积实现
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
    parser = argparse.ArgumentParser(description="v23: 大小核卷积填补 + 完整调试输出")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
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
    parser.add_argument("--conv-small-kernel", type=int, default=5, help="小核尺寸（边缘）")
    parser.add_argument("--conv-large-kernel", type=int, default=11, help="大核尺寸（内部）")
    parser.add_argument("--conv-max-iter", type=int, default=64, help="最大迭代次数")
    parser.add_argument("--left-margin-delay", type=int, default=8, help="左边界延迟处理宽度（像素）")

    # 边缘平滑 + 羽化参数
    parser.add_argument("--left-smooth-width", type=int, default=6, help="左边界垂直平滑宽度")
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5, help="左边界平滑 sigma")
    parser.add_argument("--right-blend-width", type=int, default=3, help="右边缘羽化宽度")

    return parser.parse_args()


# ==================== 核心算法：大小核卷积填补 ====================

def create_gaussian_kernel(kernel_size, device, dtype):
    """创建高斯核"""
    sigma = kernel_size / 4.0
    k = kernel_size
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_2d = gauss.view(1, k) * gauss.view(k, 1)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def inpaint_multiscale_conv(image, hole_mask, near_score, bg_threshold=0.3,
                            small_kernel=5, large_kernel=11, max_iter=64,
                            left_margin_delay=8):
    """
    大小核卷积填补 + 左边界延迟处理

    步骤：
    1. 背景掩码：只允许真正的背景像素作为种子
    2. 大核迭代：快速填充空洞内部（远离边界的区域）
    3. 小核迭代：精细处理边缘区域
    4. 左边界最后处理：防止前景颜色扩散

    Args:
        left_margin_delay: 距离左边缘多少像素以内的区域延迟到最后处理
    """
    h, w = hole_mask.shape
    device = image.device

    # 背景种子掩码（只使用真实背景像素，不用前景像素）
    bg_seed = (~hole_mask) & (near_score < bg_threshold)

    # NCHW 格式
    img_nchw = image.permute(2, 0, 1).unsqueeze(0)
    mask_nchw = bg_seed.unsqueeze(0).unsqueeze(0).float()

    # 创建大小核
    pad_small = small_kernel // 2
    pad_large = large_kernel // 2

    gauss_small = create_gaussian_kernel(small_kernel, device, image.dtype)
    gauss_large = create_gaussian_kernel(large_kernel, device, image.dtype)

    # 复制为 3 通道（每组卷积独立）
    kernel_small = gauss_small.view(1, 1, small_kernel, small_kernel).repeat(3, 1, 1, 1)
    kernel_large = gauss_large.view(1, 1, large_kernel, large_kernel).repeat(3, 1, 1, 1)

    # ========== 步骤 1：计算左边界区域（延迟处理） ==========
    edge_x = torch.full((h,), w, device=device, dtype=torch.long)
    for y in range(h):
        cols = torch.where(hole_mask[y])[0]
        if len(cols) > 0:
            edge_x[y] = cols.min()

    x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
    edge_x_2d = edge_x.view(h, 1).expand(h, w)
    left_margin = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + left_margin_delay)
    left_margin = left_margin & hole_mask

    # 分区域：普通空洞区域 vs 左边界延迟区域
    normal_hole = hole_mask & (~left_margin)

    result = img_nchw.clone()
    weight = mask_nchw.clone()

    # ========== 步骤 2：大核迭代（快速填充内部，只在普通区域） ==========
    for i in range(max_iter // 2):
        # 大核卷积
        img_conv = F.conv2d(result * weight, kernel_large, padding=pad_large, groups=3)
        weight_conv = F.conv2d(weight, kernel_large[0:1], padding=pad_large)

        # 只更新普通空洞区域的像素
        valid = (weight_conv > 1e-6) & normal_hole.unsqueeze(0).unsqueeze(0)
        if not torch.any(valid):
            break

        result = torch.where(valid, img_conv / weight_conv.clamp_min(1e-6), result)
        weight = torch.where(valid, weight_conv, weight)

    # ========== 步骤 3：小核迭代（精细处理边缘 + 左边界区域） ==========
    for i in range(max_iter // 2):
        # 小核卷积
        img_conv = F.conv2d(result * weight, kernel_small, padding=pad_small, groups=3)
        weight_conv = F.conv2d(weight, kernel_small[0:1], padding=pad_small)

        # 更新所有剩余空洞像素
        remaining_hole = (weight < 0.5) & hole_mask.unsqueeze(0).unsqueeze(0)
        if not torch.any(remaining_hole):
            break

        valid = (weight_conv > 1e-6) & remaining_hole
        result = torch.where(valid, img_conv / weight_conv.clamp_min(1e-6), result)
        weight = torch.where(valid, weight_conv, weight)

    # ========== 步骤 4：最后处理左边界区域（确保前景不污染） ==========
    # 此时内部已经填好，用小核从右向左"扫"左边缘
    for i in range(4):  # 较少迭代，只做局部平滑
        img_conv = F.conv2d(result * weight, kernel_small, padding=pad_small, groups=3)
        weight_conv = F.conv2d(weight, kernel_small[0:1], padding=pad_small)

        remaining = (weight < 0.5) & left_margin.unsqueeze(0).unsqueeze(0)
        if not torch.any(remaining):
            break

        valid = (weight_conv > 1e-6) & remaining
        result = torch.where(valid, img_conv / weight_conv.clamp_min(1e-6), result)
        weight = torch.where(valid, weight_conv, weight)

    return result[0].permute(1, 2, 0)


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


def viz_disparity(disp, max_disp=24.0):
    """视差可视化"""
    norm = (disp / max_disp).clamp(0.0, 1.0).cpu().numpy()
    norm_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def viz_mask_on_image(img_tensor, mask, color):
    """在图像上标记掩码区域"""
    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
    mask_np = mask.cpu().numpy()
    img_np[mask_np] = color
    return img_np


def create_video_writer(ffmpeg_bin, output_path, fps, w, h):
    """libx264 CPU 编码"""
    pix_fmt = "yuv420p" if (w % 2 == 0 and h % 2 == 0) else "yuv444p"
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", pix_fmt, "-movflags", "+faststart",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def process_frame_full_debug(frame_bgr, model, device, args):
    """处理单帧，返回所有调试阶段"""
    h_orig, w_orig = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).to(device).float() / 255.0

    # ========== 深度推理 ==========
    img = torch.from_numpy(frame_rgb).to(device).permute(2, 0, 1).float() / 255.0
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

    # ========== B版本阶段 ==========

    # B01: 原始视差
    b01 = viz_disparity(disparity, 24.0)

    # B02: 锐化后视差
    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity,
        kernel_size=args.sharpen_kernel_size,
        threshold=args.sharpen_threshold,
        iterations=1,
        reject_margin=args.sharpen_reject_margin
    )
    b02 = viz_disparity(disparity_sharp, 24.0)

    # B03: 被排除的过渡像素（黄色）
    b03 = viz_mask_on_image(frame_tensor, unreliable, [255, 255, 0])

    # B04 + B05: 排除过渡像素后的 warp 结果和空洞
    left_rgb = frame_tensor
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )
    b04 = (right_warped.cpu().numpy() * 255).astype(np.uint8)
    b05 = viz_mask_on_image(right_warped, hole, [255, 0, 0])

    # ========== C版本阶段 ==========

    # C06: 反遮挡带标记（绿色）
    disocclusion_band = project_disocclusion_bands_optimized(
        disparity_sharp,
        min_drop=args.band_min_drop,
        right_cleanup=args.band_right_cleanup,
        min_band_width=args.band_min_width
    )
    c06 = viz_mask_on_image(right_warped, disocclusion_band, [0, 255, 0])

    # 应用反遮挡带
    hole_with_band = hole | disocclusion_band
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    # C07: 应用反遮挡带后的空洞
    c07 = viz_mask_on_image(right_with_band, hole_with_band, [255, 0, 0])

    # 目标空间 near
    target_near = b.forward_target_near(near_score, disparity_sharp, unreliable)
    b._LAST_TARGET_NEAR = target_near

    # C08: v23 大小核卷积填补
    final_right = inpaint_multiscale_conv(
        right_with_band, hole_with_band, near_score,
        bg_threshold=0.3,
        small_kernel=args.conv_small_kernel,
        large_kernel=args.conv_large_kernel,
        max_iter=args.conv_max_iter,
        left_margin_delay=args.left_margin_delay
    )

    # ========== 左边界垂直平滑 + 右边缘羽化 ==========
    h, w = hole_with_band.shape

    # 左边界垂直平滑
    if args.left_smooth_width > 0:
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

        img_4d = final_right.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask_3d = smooth_mask.unsqueeze(-1)
        final_right = torch.where(smooth_mask_3d, img_smoothed, final_right)

    # 右边缘羽化融合
    if args.right_blend_width > 0:
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

            inpainted_colors = final_right[y, blend_x_range]
            bg_color = final_right[y, right_edge_x + 1]
            blended = alpha * inpainted_colors + (1 - alpha) * bg_color
            final_right[y, blend_x_range] = blended

    c08 = (final_right.cpu().numpy() * 255).astype(np.uint8)

    # C09: 并排对比（左=原图，右=最终结果）
    c09 = np.hstack([frame_rgb, c08])

    return {
        'b01': b01, 'b02': b02, 'b03': b03, 'b04': b04, 'b05': b05,
        'c06': c06, 'c07': c07, 'c08': c08, 'c09': c09,
        'final': c08,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v23 大小核卷积版 ✨")
    print(f"{'='*60}")
    print(f"【核心参数】")
    print(f"  锐化阈值: {args.sharpen_threshold}, 反遮挡带阈值: {args.band_min_drop}")
    print(f"  大小核: {args.conv_small_kernel}x{args.conv_small_kernel} + {args.conv_large_kernel}x{args.conv_large_kernel}")
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
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {total_frames} 帧, {fps:.2f} fps, {width}x{height}\n")

    # 创建所有调试视频 writer
    writers = {}
    writers['b01_raw_disparity'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'b01_raw_disparity.mp4'), fps, width, height
    )
    writers['b02_sharpened_disparity'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'b02_sharpened_disparity.mp4'), fps, width, height
    )
    writers['b03_excluded_pixels_yellow'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'b03_excluded_pixels_yellow.mp4'), fps, width, height
    )
    writers['b04_warped_after_exclude'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'b04_warped_after_exclude.mp4'), fps, width, height
    )
    writers['b05_hole_after_exclude_red'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'b05_hole_after_exclude_red.mp4'), fps, width, height
    )
    writers['c06_disocclusion_band_green'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'c06_disocclusion_band_green.mp4'), fps, width, height
    )
    writers['c07_hole_with_band_red'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'c07_hole_with_band_red.mp4'), fps, width, height
    )
    writers['c08_inpaint_final'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'c08_inpaint_final.mp4'), fps, width, height
    )
    writers['c09_final_stereo_sbs'] = create_video_writer(
        args.ffmpeg_bin, str(outdir / 'c09_final_stereo_sbs.mp4'), fps, width * 2, height
    )

    # 关键帧列表（用于调试分析）
    key_frames = {60, 90, 210, 240}
    key_frame_results = {}

    # 主循环
    pbar = tqdm(range(total_frames), ncols=80)
    for frame_idx in pbar:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        result = process_frame_full_debug(frame_bgr, model, device, args)

        # 保存关键帧
        if frame_idx in key_frames:
            key_frame_results[frame_idx] = result
            print(f"\n📸 保存关键帧 {frame_idx}...")

        # 写入所有阶段
        writers['b01_raw_disparity'].stdin.write(result['b01'].tobytes())
        writers['b02_sharpened_disparity'].stdin.write(result['b02'].tobytes())
        writers['b03_excluded_pixels_yellow'].stdin.write(result['b03'].tobytes())
        writers['b04_warped_after_exclude'].stdin.write(result['b04'].tobytes())
        writers['b05_hole_after_exclude_red'].stdin.write(result['b05'].tobytes())
        writers['c06_disocclusion_band_green'].stdin.write(result['c06'].tobytes())
        writers['c07_hole_with_band_red'].stdin.write(result['c07'].tobytes())
        writers['c08_inpaint_final'].stdin.write(result['c08'].tobytes())
        writers['c09_final_stereo_sbs'].stdin.write(result['c09'].tobytes())

    pbar.close()
    cap.release()

    # ========== 保存关键帧图片 ==========
    print(f"\n💾 保存关键帧截图...")
    for frame_idx, result in key_frame_results.items():
        # 全帧结果
        cv2.imwrite(str(outdir / f'v23_frame_{frame_idx:03d}_final.png'),
                   cv2.cvtColor(result['c08'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(outdir / f'v23_frame_{frame_idx:03d}_band_green.png'),
                   cv2.cvtColor(result['c06'], cv2.COLOR_RGB2BGR))

        # 手部区域裁剪
        h, w = result['c08'].shape[:2]
        x1, x2 = max(1000, 0), min(1300, w)
        y1, y2 = max(400, 0), min(700, h)
        hand_crop = cv2.cvtColor(result['c08'][y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(outdir / f'v23_frame_{frame_idx:03d}_hand_crop.png'), hand_crop)

        # 手部区域的带裁剪
        band_hand_crop = cv2.cvtColor(result['c06'][y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(outdir / f'v23_frame_{frame_idx:03d}_band_hand_crop.png'), band_hand_crop)

        # 天花板区域裁剪
        x1, x2 = max(800, 0), min(1200, w)
        y1, y2 = max(0, 0), min(250, h)
        ceil_crop = cv2.cvtColor(result['c08'][y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(outdir / f'v23_frame_{frame_idx:03d}_ceiling_crop.png'), ceil_crop)

    print(f"✅ 关键帧已保存: {sorted(key_frame_results.keys())}")

    # 关闭所有 writer
    print("\n正在关闭视频编码器...")
    for name, proc in writers.items():
        proc.stdin.close()
        proc.wait()

    print(f"\n{'='*60}")
    print(f"✅ v23 处理完成！结果保存在: {outdir}")
    print(f"\nB版本阶段（5个视频）:")
    print(f"  b01_raw_disparity.mp4           - 原始视差")
    print(f"  b02_sharpened_disparity.mp4      - 锐化后视差")
    print(f"  b03_excluded_pixels_yellow.mp4   - 黄色 = 被排除的过渡像素")
    print(f"  b04_warped_after_exclude.mp4     - warp 结果")
    print(f"  b05_hole_after_exclude_red.mp4   - 红色 = 空洞")
    print(f"\nC版本阶段（4个视频）:")
    print(f"  c06_disocclusion_band_green.mp4  - 绿色 = 反遮挡带")
    print(f"  c07_hole_with_band_red.mp4       - 应用反遮挡带后的空洞")
    print(f"  c08_inpaint_final.mp4            - 大小核卷积填补结果")
    print(f"  c09_final_stereo_sbs.mp4         - 左=原图，右=最终结果")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
