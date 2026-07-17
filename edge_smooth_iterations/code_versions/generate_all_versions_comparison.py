"""
生成所有版本的对比视频 + 统一关键帧截图
所有输出都放在同一个目录，文件名规范，方便直接对比

版本列表：
1. v22_B_original    - 原版 B 版本镜像填补（基准）
2. v29_fast_inpaint  - 大小核卷积 inpaint（对齐原版 fast_inpaint）
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

# 关键帧列表（统一）
KEY_FRAMES = [60, 90, 210, 240]


def parse_args():
    parser = argparse.ArgumentParser(description="生成所有版本对比视频 + 关键帧截图")
    parser.add_argument("--video-path", type=str, required=True, help="输入视频路径")
    parser.add_argument("--outdir", type=str, default="./frames/all_versions_comparison", help="输出目录")
    parser.add_argument("--encoder", type=str, default="vits")
    return parser.parse_args()


# ========== 反遮挡带生成（所有版本共享） ==========
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


# ========== 边缘后处理（所有版本共享：左边界平滑 + 右边缘羽化） ==========
def edge_post_process(image, hole_mask, smooth_width=6, smooth_sigma=1.5, blend_width=3):
    h, w = hole_mask.shape
    device = image.device
    result = image.clone()

    # 左边界垂直高斯平滑
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

        k = 5
        pad = k // 2
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * smooth_sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask_3d = smooth_mask.unsqueeze(-1)
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


# ========== 1. v22: B 版本镜像填补 ==========
def inpaint_v22_b_version(image, hole_mask, near_score):
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    result = b.strict_background_inpaint_gpu_b(
        image.clone(), hole_mask, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )
    return result


# ========== 2. v29: 大小核卷积（对齐原版 fast_inpaint） ==========
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

def inpaint_v29_multiscale(image, hole_mask, near_score, bg_threshold=0.3):
    if not torch.any(hole_mask):
        return image.clone()
    img = image.clone()
    h = hole_mask.clone()
    device = image.device

    # 先填边缘
    edge_mask = detect_hole_edges(h)
    img, filled = fill_edge_with_nearest_bg_fast(img, h, edge_mask, near_score, bg_threshold)
    h[filled] = False

    # 再填内部（11x11 大核）
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


# ========== 保存关键帧 ==========
def save_key_frame_crops(outdir, version_name, frame_idx, frame_bgr, hole_mask=None):
    """保存完整帧 + 手部区域 + 天花板区域"""
    # 完整帧
    cv2.imwrite(str(outdir / f'{version_name}_frame_{frame_idx:03d}_full.png'), frame_bgr)

    # 手部区域裁剪
    h, w = frame_bgr.shape[:2]
    x1, x2 = min(max(1000, 0), w-1), min(max(1300, 0), w)
    y1, y2 = min(max(400, 0), h-1), min(max(700, 0), h)
    hand = frame_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(outdir / f'{version_name}_frame_{frame_idx:03d}_hand.png'), hand)

    # 天花板区域裁剪
    x1, x2 = min(max(800, 0), w-1), min(max(1200, 0), w)
    y1, y2 = min(max(0, 0), h-1), min(max(250, 0), h)
    ceil = frame_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(outdir / f'{version_name}_frame_{frame_idx:03d}_ceiling.png'), ceil)

    # 如果有洞，可视化洞
    if hole_mask is not None:
        hole_viz = frame_bgr.copy()
        hole_np = hole_mask.cpu().numpy() if hasattr(hole_mask, 'cpu') else hole_mask
        hole_viz[hole_np] = [0, 0, 255]
        cv2.imwrite(str(outdir / f'{version_name}_frame_{frame_idx:03d}_hole_red.png'), hole_viz)


# ========== 主流程 ==========
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 生成所有版本对比 ✨")
    print(f"{'='*80}")
    print(f"版本 1: v22_B_original    - B 版本镜像填补（基准）")
    print(f"版本 2: v29_multiscale    - 大小核卷积（edge_fill_mode=1）")
    print(f"{'='*80}")
    print(f"关键帧: {KEY_FRAMES}")

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
    print(f"视频: {total_frames} 帧, {fps} fps, {width}x{height}\n")

    # 创建视频 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writers = {
        'v22_B_original': cv2.VideoWriter(str(outdir / 'v22_B_original_only.mp4'), fourcc, fps, (width, height)),
        'v29_multiscale': cv2.VideoWriter(str(outdir / 'v29_multiscale_only.mp4'), fourcc, fps, (width, height)),
        'compare_sbs': cv2.VideoWriter(str(outdir / 'all_versions_sbs_compare.mp4'), fourcc, fps, (width * 2, height)),
    }

    # 处理所有帧
    for frame_idx in tqdm(range(total_frames), ncols=80):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h_orig, w_orig = frame_bgr.shape[:2]

        # ========== 深度推理（所有版本共享） ==========
        img = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
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

        left_rgb = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        # 反遮挡带（所有版本共享）
        disocclusion_band = project_disocclusion_bands_optimized(disparity_sharp)
        hole_with_band = hole | disocclusion_band

        right_with_band = right_warped.clone()
        right_with_band[hole_with_band] = 0.0

        # ========== 版本 1: v22 B 版本镜像填补 ==========
        result_v22 = inpaint_v22_b_version(right_with_band, hole_with_band, near_score)
        result_v22 = edge_post_process(result_v22, hole_with_band)
        v22_u8 = (result_v22.cpu().numpy() * 255).astype(np.uint8)
        v22_bgr = cv2.cvtColor(v22_u8, cv2.COLOR_RGB2BGR)
        writers['v22_B_original'].write(v22_bgr)

        # ========== 版本 2: v29 大小核卷积 ==========
        result_v29 = inpaint_v29_multiscale(right_with_band, hole_with_band, near_score)
        result_v29 = edge_post_process(result_v29, hole_with_band)
        v29_u8 = (result_v29.cpu().numpy() * 255).astype(np.uint8)
        v29_bgr = cv2.cvtColor(v29_u8, cv2.COLOR_RGB2BGR)
        writers['v29_multiscale'].write(v29_bgr)

        # ========== 并排对比视频 ==========
        sbs_frame = np.hstack([v22_bgr, v29_bgr])
        writers['compare_sbs'].write(sbs_frame)

        # ========== 关键帧保存 ==========
        if frame_idx in KEY_FRAMES:
            print(f"\n📸 保存关键帧 {frame_idx}...")
            save_key_frame_crops(outdir, 'v22_B', frame_idx, v22_bgr, hole_with_band)
            save_key_frame_crops(outdir, 'v29_multiscale', frame_idx, v29_bgr)
            cv2.imwrite(str(outdir / f'frame_{frame_idx:03d}_original_left.png'), frame_bgr)

    cap.release()
    for w in writers.values():
        w.release()

    print(f"\n{'='*80}")
    print(f"✅ 所有版本已生成完毕！")
    print(f"输出目录: {outdir}/")
    print(f"\n📁 文件清单:")
    for f in sorted(outdir.glob('*.mp4')):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"\n🖼️  关键帧截图:")
    for frame_idx in KEY_FRAMES:
        print(f"  帧 {frame_idx}:")
        print(f"    - *_full.png        - 完整帧")
        print(f"    - *_hand.png        - 手部区域")
        print(f"    - *_ceiling.png     - 天花板区域")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
