"""
迭代 v26: 极致速度版本 - 单大核 + 极少迭代 ✨

目标：单帧填补 < 10ms

优化：
1. 只用 15x15 大单核，不区分边缘/非边缘（更少的kernel launch）
2. 极少迭代次数（8-12次）
3. 不单独检测边缘，直接卷积
4. 更激进的提前终止
"""
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, '.')
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_KERNEL_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="v26: 极致速度 - 单大核 + 少迭代")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")

    parser.add_argument("--sharpen-threshold", type=float, default=3.0)
    parser.add_argument("--band-min-drop", type=float, default=3.5)
    parser.add_argument("--band-min-width", type=int, default=8)

    parser.add_argument("--kernel-size", type=int, default=15, help="单一大核尺寸")
    parser.add_argument("--max-iter", type=int, default=10, help="最大迭代次数（更少）")

    parser.add_argument("--left-smooth-width", type=int, default=6)
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5)
    parser.add_argument("--right-blend-width", type=int, default=3)

    return parser.parse_args()


def get_kernel(kernel_size, device, dtype):
    """缓存 uniform kernel"""
    key = (kernel_size, str(device), str(dtype))
    if key not in _KERNEL_CACHE:
        kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype) / (kernel_size * kernel_size)
        kernel3 = kernel1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[key] = (kernel1, kernel3)
    return _KERNEL_CACHE[key]


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
def inpaint_ultrafast(image, hole_mask, near_score, bg_threshold=0.3,
                      kernel_size=15, max_iter=10):
    """
    极致速度：单大核 + 极少迭代

    设计：用 15x15 大核，每次能填补距离边缘 7 像素的洞
    10 次迭代理论上能填 70 像素宽的洞（足够）
    """
    h, w = hole_mask.shape
    device = image.device

    is_bg = (near_score < bg_threshold).to(image.dtype).unsqueeze(0).unsqueeze(0)

    img = image.clone()
    hole = hole_mask.clone()
    if not torch.any(hole):
        return img

    pad = kernel_size // 2
    kernel1, kernel3 = get_kernel(kernel_size, device, img.dtype)

    img_nchw = img.permute(2, 0, 1).unsqueeze(0).contiguous()

    for i in range(max_iter):
        if not torch.any(hole):
            break

        known = (~hole).float().unsqueeze(0).unsqueeze(0)
        weight = known * is_bg

        # 单次大核卷积
        count = F.conv2d(weight, kernel1, padding=pad)
        fillable = hole & (count[0, 0] > 0.01)

        if not torch.any(fillable):
            break

        rgb_sum = F.conv2d(img_nchw * weight, kernel3, padding=pad, groups=3)
        avg = rgb_sum / count.clamp_min(1e-6)

        fill_mask_3d = fillable.unsqueeze(-1).expand_as(img)
        img[fill_mask_3d] = avg[0].permute(1, 2, 0)[fill_mask_3d]
        hole[fillable] = False

        if torch.any(~fillable):
            img_nchw = img.permute(2, 0, 1).unsqueeze(0).contiguous()

    return img


def edge_post_process(image, hole_mask,
                      smooth_width=6, smooth_sigma=1.5,
                      blend_width=3):
    """边缘后处理"""
    h, w = hole_mask.shape
    device = image.device

    if smooth_width > 0:
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        smooth_mask = torch.zeros_like(hole_mask)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + smooth_width)
        smooth_mask = smooth_region & hole_mask

        k = 5
        pad = k // 2
        sigma = smooth_sigma
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = image.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask_3d = smooth_mask.unsqueeze(-1)
        image = torch.where(smooth_mask_3d, img_smoothed, image)

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

            inpainted_colors = image[y, blend_x_range]
            bg_color = image[y, right_edge_x + 1]
            blended = alpha * inpainted_colors + (1 - alpha) * bg_color
            image[y, blend_x_range] = blended

    return image


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v26 极致速度 - 单大核 ✨")
    print(f"{'='*60}")
    print(f"  目标: 单帧填补 < 10ms")
    print(f"  核: {args.kernel_size}x{args.kernel_size}, 迭代: {args.max_iter}")
    print(f"{'='*60}\n")

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
    print(f"视频信息: {total_frames} 帧, {fps} fps, {width}x{height}\n")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_opt = cv2.VideoWriter(str(outdir / 'v26_ultrafast.mp4'), fourcc, fps, (width, height))

    import time
    from tqdm import tqdm

    inpaint_times = []
    for frame_idx in tqdm(range(total_frames), ncols=80):
        ok, frame_bgr = cap.read()
        if not ok:
            break

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
            disparity, kernel_size=15, threshold=args.sharpen_threshold,
            iterations=1, reject_margin=0.10
        )

        left_rgb = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        disocclusion_band = project_disocclusion_bands_optimized(
            disparity_sharp,
            min_drop=args.band_min_drop,
            min_band_width=args.band_min_width
        )
        hole_with_band = hole | disocclusion_band

        right_with_band = right_warped.clone()
        right_with_band[hole_with_band] = 0.0

        # ========== 计时开始 ==========
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        final_right = inpaint_ultrafast(
            right_with_band, hole_with_band, near_score,
            bg_threshold=0.3,
            kernel_size=args.kernel_size,
            max_iter=args.max_iter
        )

        final_right = edge_post_process(
            final_right, hole_with_band,
            smooth_width=args.left_smooth_width,
            smooth_sigma=args.left_smooth_sigma,
            blend_width=args.right_blend_width
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inpaint_times.append((t1 - t0) * 1000)

        opt_u8 = (final_right.cpu().numpy() * 255).astype(np.uint8)
        opt_bgr = cv2.cvtColor(opt_u8, cv2.COLOR_RGB2BGR)
        out_opt.write(opt_bgr)

    cap.release()
    out_opt.release()

    avg_time = np.mean(inpaint_times)
    p95_time = np.percentile(inpaint_times, 95)
    max_time = np.max(inpaint_times)

    print(f"\n{'='*60}")
    print(f"✅ v26 处理完成！结果保存在: {outdir}")
    print(f"\n⏱️  填补时间统计:")
    print(f"  平均: {avg_time:.2f} ms")
    print(f"  P95:  {p95_time:.2f} ms")
    print(f"  最大: {max_time:.2f} ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
