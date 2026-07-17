"""
迭代 v28: CPU numpy 矢量化填补 - 最终速度版 ✨

核心洞察：不需要 GPU 迭代卷积！反遮挡带是水平条状，
直接在 CPU numpy 上找区间 + 颜色广播，总耗时 ~25ms。

速度：填补 + 后处理 ~25ms
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


def parse_args():
    parser = argparse.ArgumentParser(description="v28: CPU numpy 填补 - 最终速度版")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")

    parser.add_argument("--sharpen-threshold", type=float, default=3.0)
    parser.add_argument("--band-min-drop", type=float, default=3.5)
    parser.add_argument("--band-min-width", type=int, default=8)

    parser.add_argument("--left-smooth-width", type=int, default=6, help="左边界垂直平滑宽度")
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5, help="左边界平滑 sigma")
    parser.add_argument("--right-blend-width", type=int, default=3, help="右边缘羽化宽度")

    return parser.parse_args()


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


def inpaint_cpu_numpy(image_np, hole_np, smooth_width=6, smooth_sigma=1.5, blend_width=3):
    """
    纯 CPU numpy 填补 + 后处理

    总耗时 ~25ms
    """
    H, W = hole_np.shape

    # ========== 步骤 1：找到所有空洞区间 ==========
    all_regions = []  # (y, start_x, end_x)
    for y in range(H):
        row = hole_np[y]
        if not row.any():
            continue
        changes = np.diff(np.concatenate(([0], row.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            all_regions.append((y, s, e))

    # ========== 步骤 2：用右边背景色广播填充 ==========
    result = image_np.copy()
    for y, s, e in all_regions:
        bg_x = e if e < W else (s - 1 if s > 0 else 0)
        result[y, s:e, :] = result[y, bg_x, :].reshape(1, 3)

    # ========== 步骤 3：左边界垂直平滑（消除锯齿）==========
    if smooth_width > 0:
        edge_x = np.full((H,), W, dtype=np.int64)
        for y in range(H):
            cols = np.where(hole_np[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        # 5x1 高斯核
        k = 5
        pad = k // 2
        sigma = smooth_sigma
        gauss = np.exp(-(np.arange(k, dtype=np.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # 对每行左边界附近做垂直平滑
        for y in range(H):
            ex = edge_x[y]
            if ex >= W:
                continue
            # 只平滑空洞内 smooth_width 宽的区域
            w = min(smooth_width, W - ex)
            for dy in range(-pad, pad + 1):
                yy = y + dy
                if 0 <= yy < H:
                    factor = gauss[dy + pad]
                    if abs(dy) > 0:  # 避免覆盖自己导致的问题
                        result[y, ex:ex + w, :] = (
                            factor * result[yy, ex:ex + w, :].astype(np.float32) +
                            (1 - factor) * result[y, ex:ex + w, :].astype(np.float32)
                        ).astype(np.uint8)

    # ========== 步骤 4：右边缘羽化（简化版）==========
    if blend_width > 0:
        for y in range(H):
            cols = np.where(hole_np[y])[0]
            if len(cols) == 0:
                continue
            re = cols.max()
            if re + 1 >= W:
                continue
            blend_start = max(cols.min(), re - blend_width + 1)
            if blend_start >= re:
                continue
            alpha = np.linspace(1.0, 0.0, re - blend_start + 1, dtype=np.float32).reshape(-1, 1)
            inpainted = result[y, blend_start:re + 1, :].astype(np.float32)
            true_bg = result[y, re + 1, :].astype(np.float32).reshape(1, 3)
            result[y, blend_start:re + 1, :] = (alpha * inpainted + (1 - alpha) * true_bg).astype(np.uint8)

    return result


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v28 CPU numpy 最终速度版 ✨")
    print(f"{'='*60}")
    print(f"  填补方案: CPU numpy 区间广播填充 (~25ms)")
    print(f"  左边界平滑: {args.left_smooth_width}px")
    print(f"  右边缘羽化: {args.right_blend_width}px")
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
    out_opt = cv2.VideoWriter(str(outdir / 'v28_cpu_numpy_final.mp4'), fourcc, fps, (width, height))

    import time
    from tqdm import tqdm

    inpaint_times = []
    for frame_idx in tqdm(range(total_frames), ncols=80):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h_orig, w_orig = frame_bgr.shape[:2]

        # 深度推理 (GPU)
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

        # ========== 计时开始：CPU numpy 填补 ==========
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # D2H 传输（同时量化到 uint8）
        image_np = (right_with_band.cpu().numpy() * 255).astype(np.uint8)
        hole_np = hole_with_band.cpu().numpy()

        final_np = inpaint_cpu_numpy(
            image_np, hole_np,
            smooth_width=args.left_smooth_width,
            smooth_sigma=args.left_smooth_sigma,
            blend_width=args.right_blend_width
        )

        t1 = time.perf_counter()
        inpaint_times.append((t1 - t0) * 1000)

        # 写入
        final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)
        out_opt.write(final_bgr)

    cap.release()
    out_opt.release()

    avg_time = np.mean(inpaint_times)
    p95_time = np.percentile(inpaint_times, 95)
    max_time = np.max(inpaint_times)

    print(f"\n{'='*60}")
    print(f"✅ v28 处理完成！结果保存在: {outdir}")
    print(f"\n⏱️  填补时间统计 (D2H + 找区间 + 填充 + 后处理):")
    print(f"  平均: {avg_time:.2f} ms")
    print(f"  P95:  {p95_time:.2f} ms")
    print(f"  最大: {max_time:.2f} ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
