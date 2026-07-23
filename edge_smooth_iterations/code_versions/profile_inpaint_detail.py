"""
Inpaint 阶段深度性能分析
分析每次迭代的耗时，以及 conv2d 各参数的影响
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


def create_asymmetric_kernel(kernel_size, device, left_width=3):
    pad = kernel_size // 2
    x_indices = torch.arange(kernel_size, device=device) - pad
    horizontal_mask = x_indices >= -left_width
    horizontal_weights = torch.ones(kernel_size, device=device, dtype=torch.float32)
    horizontal_weights[~horizontal_mask] = 0.0
    kernel_1d = horizontal_weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    num_nonzero = horizontal_weights.sum().item() * kernel_size
    if num_nonzero > 0:
        kernel_2d = kernel_2d / kernel_2d.sum() * num_nonzero
    return kernel_2d


@torch.no_grad()
def inpaint_with_timing(img, hole, kernel, k3, max_iter):
    """带逐次迭代计时的 inpaint"""
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2

    iter_times = []
    conv_times = []
    fill_counts = []

    for it in range(max_iter):
        t0 = time.perf_counter()

        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

        t1 = time.perf_counter()
        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, kernel, padding=pad)
        torch.cuda.synchronize()
        t_conv = (time.perf_counter() - t1) * 1000
        conv_times.append(t_conv)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
        filled = can_fill.sum().item()
        fill_counts.append(filled)

        if filled == 0:
            break

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        torch.cuda.synchronize()
        t_iter = (time.perf_counter() - t0) * 1000
        iter_times.append(t_iter)

        if not hole_cur.any():
            break

    return iter_times, conv_times, fill_counts, hole_cur.sum().item()


def test_kernel_sizes(img, hole, device):
    """测试不同 kernel size 的性能"""
    print(f"\n{'='*60}")
    print(f"【不同 kernel size 性能对比】")
    print(f"{'='*60}")
    print(f"{'Kernel Size':>12} {'30 iter总ms':>12} {'每次ms':>10} {'剩余空洞':>12}")
    print("-" * 60)

    for ks in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        kernel = create_asymmetric_kernel(ks, device, 3)
        k3 = kernel.repeat(3, 1, 1, 1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iter_times, conv_times, fill_counts, remaining = inpaint_with_timing(
            img.clone(), hole.clone(), kernel, k3, max_iter=30
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        print(f"{ks:>12} {total_ms:>12.1f} {total_ms/max(1,len(iter_times)):>10.1f} {remaining:>12,d}")


def test_left_widths(img, hole, device):
    """测试不同 left_width 的性能和填充效果"""
    print(f"\n{'='*60}")
    print(f"【不同 left_width 性能对比 (kernel=15)】")
    print(f"{'='*60}")
    print(f"{'Left Width':>12} {'30 iter总ms':>12} {'每次ms':>10} {'剩余空洞':>12}")
    print("-" * 60)

    for lw in [0, 1, 2, 3, 4, 5, 7, 15]:
        kernel = create_asymmetric_kernel(15, device, lw)
        k3 = kernel.repeat(3, 1, 1, 1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iter_times, conv_times, fill_counts, remaining = inpaint_with_timing(
            img.clone(), hole.clone(), kernel, k3, max_iter=30
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000

        print(f"{lw:>12} {total_ms:>12.1f} {total_ms/max(1,len(iter_times)):>10.1f} {remaining:>12,d}")


def test_max_iter(img, hole, device):
    """测试不同 max_iter 的收敛情况"""
    print(f"\n{'='*60}")
    print(f"【迭代收敛分析 (kernel=15, left_width=3)】")
    print(f"{'='*60}")
    print(f"{'迭代次数':>8} {'累计ms':>10} {'本轮填充':>12} {'剩余空洞':>12}")
    print("-" * 55)

    kernel = create_asymmetric_kernel(15, device, 3)
    k3 = kernel.repeat(3, 1, 1, 1)

    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2

    total_ms = 0
    for it in range(50):
        t0 = time.perf_counter()

        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, kernel, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
        filled = can_fill.sum().item()

        if filled == 0:
            break

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000
        remaining = hole_cur.sum().item()

        if it < 10 or (it + 1) % 5 == 0 or remaining == 0:
            print(f"{it+1:>8} {total_ms:>10.1f} {filled:>12,d} {remaining:>12,d}")

        if remaining == 0:
            break


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # 加载一帧
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    h, w = frame_bgr.shape[:2]
    print(f"帧尺寸: {w}×{h} = {w*h:,} 像素\n")

    # 预处理
    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    input_size = 518
    scale = input_size / max(h, w)
    depth_h = max(14, int(round(h * scale / 14)) * 14)
    depth_w = max(14, int(round(w * scale / 14)) * 14)
    if w >= h:
        depth_w = input_size
    else:
        depth_h = input_size

    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    img_resized = F.interpolate(img, size=(depth_h, depth_w), mode="bilinear", align_corners=False)
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
        size=(h, w),
        mode="bilinear", align_corners=False
    )[0, 0]
    disparity = near_score * 24.0

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    # 膨胀
    hole_float = hole.float()
    cumsum_from_right = torch.cumsum(hole_float.flip(dims=[1]), dim=1).flip(dims=[1])
    hole_widths = cumsum_from_right * hole_float
    dilate_per_pixel = (hole_widths / 24.0 * 5).round().long()
    dilate_per_pixel = torch.clamp(dilate_per_pixel, 1, 5)
    hole_dilated = hole.clone()
    for shift in range(1, 5 + 1):
        shifted = torch.roll(hole, shifts=shift, dims=1)
        shifted[:, :shift] = False
        should_dilate = (dilate_per_pixel >= shift) & shifted
        hole_dilated = hole_dilated | should_dilate

    right_with_hole = right_warped.clone()
    right_with_hole[hole_dilated] = 0.0

    print(f"原始空洞: {hole.sum().item():,} 像素")
    print(f"膨胀后空洞: {hole_dilated.sum().item():,} 像素")
    print(f"空洞占比: {hole_dilated.sum().item()/(w*h)*100:.2f}%")

    # 运行测试
    test_kernel_sizes(right_with_hole, hole_dilated, device)
    test_left_widths(right_with_hole, hole_dilated, device)
    test_max_iter(right_with_hole, hole_dilated, device)


if __name__ == "__main__":
    main()
