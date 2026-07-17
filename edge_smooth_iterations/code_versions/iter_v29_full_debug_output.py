"""
v29 完整调试输出 - 和 v23 目录格式 100% 一致
方便你直接和 v23 做逐帧对比研究

输出内容完全对应：
b01_raw_disparity.mp4          - 原始视差
b02_sharpened_disparity.mp4   - 锐化后视差
b03_excluded_pixels_yellow.mp4 - 黄色=被排除的过渡像素
b04_warped_after_exclude.mp4   - warp 结果
b05_hole_after_exclude_red.mp4 - 红色=空洞
c06_disocclusion_band_green.mp4 - 绿色=反遮挡带
c07_hole_with_band_red.mp4     - 红色=空洞+带
c08_inpaint_final.mp4          - 大小核填补结果
c09_final_stereo_sbs.mp4       - 左=原图, 右=结果

关键帧 (60, 90, 210, 240):
  v29_frame_xxx_final.png
  v29_frame_xxx_band_green.png
  v29_frame_xxx_hand_crop.png
  v29_frame_xxx_band_hand_crop.png
  v29_frame_xxx_ceiling_crop.png
"""
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, '.')
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
KEY_FRAMES = [60, 90, 210, 240]


def parse_args():
    parser = argparse.ArgumentParser(description="v29: 大小核卷积 - 完整调试输出（与v23完全对应）")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./frames/v29_multiscale_final")
    parser.add_argument("--encoder", type=str, default="vits")
    return parser.parse_args()


def disparity_to_color(disp_tensor, max_disp=24.0):
    """视差可视化 -> RGB uint8 HWC"""
    norm = (disp_tensor / max_disp).clamp(0.0, 1.0).cpu().numpy()
    norm_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


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
    start, end = start.clamp(0, w - 1), end.clamp(0, w - 1)
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


_KERNEL_CACHE = {}
def get_kernel(k, device, dtype_str):
    key = (k, str(device), dtype_str)
    if key not in _KERNEL_CACHE:
        kernel1 = torch.ones((1, 1, k, k), device=device, dtype=getattr(torch, dtype_str)) / (k * k)
        kernel3 = kernel1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[key] = (kernel1, kernel3)
    return _KERNEL_CACHE[key]


def detect_hole_edges(hole_mask):
    is_hole = hole_mask.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones((1, 1, 3, 3), device=hole_mask.device, dtype=is_hole.dtype)
    neighbor_count = F.conv2d(1 - is_hole, kernel, padding=1)
    edge_mask = hole_mask & (neighbor_count[0, 0] > 0.01)
    return edge_mask


def fill_edge_with_nearest_bg_fast(img, hole, edge_mask, near, bg_threshold=0.3):
    result = img.clone()
    filled_mask = torch.zeros_like(hole)
    bg_mask = ~hole & (near < bg_threshold)
    if torch.any(edge_mask):
        edge_extended = F.max_pool2d(
            edge_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=5, stride=1, padding=2
        )[0, 0] > 0
        to_fill = edge_extended & hole
        if torch.any(to_fill):
            img_nchw = img.permute(2, 0, 1).unsqueeze(0)
            bg_mask_nchw = bg_mask.unsqueeze(0).unsqueeze(0).float()
            kernel1, kernel3 = get_kernel(5, img.device, str(img.dtype).split('.')[-1])
            rgb_sum = F.conv2d(img_nchw * bg_mask_nchw, kernel3, padding=2, groups=3)
            weight_sum = F.conv2d(bg_mask_nchw, kernel1, padding=2)
            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)
            valid_fill = to_fill & (weight_sum[0, 0] > 0.5)
            if torch.any(valid_fill):
                result[valid_fill] = avg_hwc[valid_fill]
                filled_mask[valid_fill] = True
    return result, filled_mask


def inpaint_multiscale(image, hole_mask, near_score, bg_threshold=0.3):
    if not torch.any(hole_mask):
        return image.clone()
    img = image.clone()
    h = hole_mask.clone()
    device = image.device

    # edge_fill_mode=1: 先填边缘
    edge_mask = detect_hole_edges(h)
    img, filled = fill_edge_with_nearest_bg_fast(img, h, edge_mask, near_score, bg_threshold)
    h[filled] = False

    # 大核填内部
    kernel11, kernel11_3 = get_kernel(11, device, str(img.dtype).split('.')[-1])
    while h.any():
        known = (~h).float().unsqueeze(0).unsqueeze(0)
        count = F.conv2d(known, kernel11, padding=5)
        fillable = h & (count[0, 0] > 0.01)
        if not fillable.any():
            break
        img_nchw = img.permute(2, 0, 1).unsqueeze(0)
        rgb_sum = F.conv2d(img_nchw * known, kernel11_3, padding=5, groups=3)
        avg = rgb_sum / count.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)
        img[fillable] = avg_hwc[fillable]
        h[fillable] = False

    return img


def edge_post_process(image, hole_mask, smooth_width=6, smooth_sigma=1.5, blend_width=3):
    h, w = hole_mask.shape
    device = image.device
    result = image.clone()

    if smooth_width > 0:
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        smooth_mask = torch.zeros_like(hole_mask)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + smooth_width) & hole_mask

        k, pad = 5, 2
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * smooth_sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask_3d = smooth_mask.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

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


def save_key_frames(outdir, frame_idx, orig_bgr, warped_rgb_u8, band_mask, final_rgb_u8):
    """保存 v23 完全一致的 5 张关键帧截图"""
    h, w = orig_bgr.shape[:2]

    # final.png
    final_bgr = cv2.cvtColor(final_rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(outdir / f'v29_frame_{frame_idx:03d}_final.png'), final_bgr)

    # band_green.png
    band_viz = warped_rgb_u8.copy()
    band_viz[band_mask.cpu().numpy()] = [0, 255, 0]
    band_viz_bgr = cv2.cvtColor(band_viz, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(outdir / f'v29_frame_{frame_idx:03d}_band_green.png'), band_viz_bgr)

    # hand_crop.png (1000-1300, 400-700)
    x1, x2 = min(1000, w-1), min(1300, w)
    y1, y2 = min(400, h-1), min(700, h)
    hand_crop = final_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(outdir / f'v29_frame_{frame_idx:03d}_hand_crop.png'), hand_crop)

    # band_hand_crop.png
    band_hand_crop = band_viz_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(outdir / f'v29_frame_{frame_idx:03d}_band_hand_crop.png'), band_hand_crop)

    # ceiling_crop.png (800-1200, 0-250)
    x1, x2 = min(800, w-1), min(1200, w)
    y1, y2 = 0, min(250, h)
    ceil_crop = final_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(outdir / f'v29_frame_{frame_idx:03d}_ceiling_crop.png'), ceil_crop)


def viz_mask_on_image(img_rgb_u8, mask, color):
    """把 mask 标上颜色"""
    viz = img_rgb_u8.copy()
    viz[mask.cpu().numpy()] = color
    return viz


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v29 大小核卷积 - 完整调试输出 ✨")
    print(f"{'='*80}")
    print(f"输出目录: {args.outdir}/")
    print(f"输出内容: 9 个调试视频 + 4 个关键帧 × 5 张截图")
    print(f"关键帧: {KEY_FRAMES}")
    print(f"{'='*80}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {total_frames} 帧, {fps} fps, {width}x{height}\n")

    # 9 个视频 writer（和 v23 命名完全一致）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {
        'b01': cv2.VideoWriter(str(outdir / 'b01_raw_disparity.mp4'), fourcc, fps, (width, height)),
        'b02': cv2.VideoWriter(str(outdir / 'b02_sharpened_disparity.mp4'), fourcc, fps, (width, height)),
        'b03': cv2.VideoWriter(str(outdir / 'b03_excluded_pixels_yellow.mp4'), fourcc, fps, (width, height)),
        'b04': cv2.VideoWriter(str(outdir / 'b04_warped_after_exclude.mp4'), fourcc, fps, (width, height)),
        'b05': cv2.VideoWriter(str(outdir / 'b05_hole_after_exclude_red.mp4'), fourcc, fps, (width, height)),
        'c06': cv2.VideoWriter(str(outdir / 'c06_disocclusion_band_green.mp4'), fourcc, fps, (width, height)),
        'c07': cv2.VideoWriter(str(outdir / 'c07_hole_with_band_red.mp4'), fourcc, fps, (width, height)),
        'c08': cv2.VideoWriter(str(outdir / 'c08_inpaint_final.mp4'), fourcc, fps, (width, height)),
        'c09': cv2.VideoWriter(str(outdir / 'c09_final_stereo_sbs.mp4'), fourcc, fps, (width * 2, height)),
    }

    key_frame_data = {}  # 缓存关键帧

    for frame_idx in tqdm(range(total_frames), ncols=80):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h_orig, w_orig = frame_bgr.shape[:2]
        orig_rgb_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(orig_rgb_u8).to(device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
        img_resized = F.interpolate(img, size=(294, 518), mode="bilinear", align_corners=False)
        mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        model_input = (img_resized - mean_t) / std_t

        with torch.no_grad():
            depth_raw = model(model_input)[0].float()

        flat = depth_raw.reshape(-1)
        idx = torch.randint(0, flat.numel(), (16384,), device=flat.device)
        sample = flat[idx]
        q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=flat.device))
        low, high = q_vals[0], q_vals[1]
        depth_norm = ((depth_raw - low) / (high - low)).clamp(0.0, 1.0)

        near_score = F.interpolate(
            depth_norm[None, None, :, :],
            size=(h_orig, w_orig),
            mode="bilinear", align_corners=False
        )[0, 0]

        disparity = near_score * 24.0
        disparity_sharp, unreliable = b.sharpen_disparity_edges(
            disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
        )

        left_rgb = torch.from_numpy(orig_rgb_u8).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        disocclusion_band = project_disocclusion_bands_optimized(disparity_sharp)
        hole_with_band = hole | disocclusion_band

        right_with_band = right_warped.clone()
        right_with_band[hole_with_band] = 0.0

        # ========== 大小核填补 ==========
        inpainted = inpaint_multiscale(right_with_band, hole_with_band, near_score)
        inpainted = edge_post_process(inpainted, hole_with_band)

        # ========== 生成所有阶段可视化 ==========
        b01_rgb = disparity_to_color(disparity, 24.0)
        b02_rgb = disparity_to_color(disparity_sharp, 24.0)

        warped_rgb_u8 = (right_warped.cpu().numpy() * 255).astype(np.uint8)

        b03_rgb = viz_mask_on_image(orig_rgb_u8, unreliable, [255, 255, 0])  # 左图 + 黄色
        b05_rgb = viz_mask_on_image(warped_rgb_u8, hole, [255, 0, 0])           # 红色空洞
        c06_rgb = viz_mask_on_image(warped_rgb_u8, disocclusion_band, [0, 255, 0])  # 绿色带
        c07_rgb = viz_mask_on_image(warped_rgb_u8, hole_with_band, [255, 0, 0])   # 红色空洞+带

        c08_rgb_u8 = (inpainted.cpu().numpy() * 255).astype(np.uint8)

        # 写入视频
        writers['b01'].write(cv2.cvtColor(b01_rgb, cv2.COLOR_RGB2BGR))
        writers['b02'].write(cv2.cvtColor(b02_rgb, cv2.COLOR_RGB2BGR))
        writers['b03'].write(cv2.cvtColor(b03_rgb, cv2.COLOR_RGB2BGR))
        writers['b04'].write(cv2.cvtColor(warped_rgb_u8, cv2.COLOR_RGB2BGR))
        writers['b05'].write(cv2.cvtColor(b05_rgb, cv2.COLOR_RGB2BGR))
        writers['c06'].write(cv2.cvtColor(c06_rgb, cv2.COLOR_RGB2BGR))
        writers['c07'].write(cv2.cvtColor(c07_rgb, cv2.COLOR_RGB2BGR))
        writers['c08'].write(cv2.cvtColor(c08_rgb_u8, cv2.COLOR_RGB2BGR))

        c09_sbs = np.hstack([orig_rgb_u8, c08_rgb_u8])
        writers['c09'].write(cv2.cvtColor(c09_sbs, cv2.COLOR_RGB2BGR))

        # ========== 缓存关键帧数据 ==========
        if frame_idx in KEY_FRAMES:
            key_frame_data[frame_idx] = (
                frame_bgr.copy(), warped_rgb_u8.copy(), disocclusion_band.clone(), c08_rgb_u8.copy()
            )

    cap.release()
    for w in writers.values():
        w.release()

    # ========== 保存关键帧截图 ==========
    print(f"\n📸 正在保存关键帧...")
    for frame_idx in KEY_FRAMES:
        if frame_idx in key_frame_data:
            orig_bgr, warped_rgb_u8, band, final_rgb_u8 = key_frame_data[frame_idx]
            save_key_frames(outdir, frame_idx, orig_bgr, warped_rgb_u8, band, final_rgb_u8)
            print(f"   帧 {frame_idx}: ✅")

    print(f"\n{'='*80}")
    print(f"✅ v29 完整输出已生成！")
    print(f"输出目录: {outdir}/")
    print(f"\n📁 视频文件 (9个):")
    for f in sorted(outdir.glob('*.mp4')):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"\n🖼️  关键帧截图 ({len(KEY_FRAMES)} 帧 × 5 张 = {len(KEY_FRAMES)*5} 张 PNG):")
    for frame_idx in KEY_FRAMES:
        print(f"  帧 {frame_idx}: 5 张")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
