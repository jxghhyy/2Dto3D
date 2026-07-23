"""
v47 全过程耗时详细拆解
分析每帧各阶段的时间分布，找出瓶颈
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
NUM_FRAMES = 200  # 只测前 200 帧


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
def dilate_hole_right_adaptive_gpu(hole, max_dilate, max_disparity, device):
    h, w = hole.shape
    hole_float = hole.float()
    cumsum_from_right = torch.cumsum(hole_float.flip(dims=[1]), dim=1).flip(dims=[1])
    hole_widths = cumsum_from_right * hole_float
    dilate_per_pixel = (hole_widths / max_disparity * max_dilate).round().long()
    dilate_per_pixel = torch.clamp(dilate_per_pixel, 1, max_dilate)
    dilated = hole.clone()
    for shift in range(1, max_dilate + 1):
        shifted = torch.roll(hole, shifts=shift, dims=1)
        shifted[:, :shift] = False
        should_dilate = (dilate_per_pixel >= shift) & shifted
        dilated = dilated | should_dilate
    return dilated


@torch.no_grad()
def inpaint_with_kernel(img, hole, kernel, k3, max_iter):
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2
    for it in range(max_iter):
        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask
        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)
        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, kernel, padding=pad)
        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)
        can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
        if can_fill.sum().item() == 0:
            break
        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False
        if not hole_cur.any():
            break
    return result, hole_cur


def main():
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v47 Profiling] 设备: {device}")
    print(f"[v47 Profiling] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"[v47 Profiling] 测试前 {NUM_FRAMES} 帧\n")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    print("预热中...")
    dummy = torch.randn(1, 3, 518, 518, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    torch.cuda.synchronize()
    print("预热完成\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {w}×{h}, {fps} fps\n")

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
    max_disparity = 24.0

    kernel = create_asymmetric_kernel(15, device, 3)
    k3 = kernel.repeat(3, 1, 1, 1)

    stage_times = {
        "0_read_frame": [],
        "1_bgr_to_rgb": [],
        "2_resize_img": [],
        "3_normalize": [],
        "4_depth_infer": [],
        "5_depth_normalize": [],
        "6_resize_depth": [],
        "7_sharpen_edges": [],
        "8_gpu_warp": [],
        "9_dilate_hole": [],
        "A_inpaint_kernel": [],
        "B_to_cpu_bgr": [],
        "C_write_video": [],
    }

    frame_times = []
    hole_counts = []
    inpaint_iters = []

    print(f"{'帧':>5} {'总ms':>8} {'深度':>8} {'Warp':>8} {'Inpaint':>8} {'空洞':>8}")
    print("-" * 60)

    for frame_idx in range(NUM_FRAMES):
        t_frame_start = time.perf_counter()
        t_stage = {}

        # 0. 读帧
        t0 = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t_stage["0_read_frame"] = (time.perf_counter() - t0) * 1000

        # 1. BGR to RGB
        t0 = time.perf_counter()
        left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t_stage["1_bgr_to_rgb"] = (time.perf_counter() - t0) * 1000

        # 2. Image to tensor
        t0 = time.perf_counter()
        img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
        img_resized = F.interpolate(img, size=(depth_h, depth_w), mode="bilinear", align_corners=False)
        t_stage["2_resize_img"] = (time.perf_counter() - t0) * 1000

        # 3. Normalize
        t0 = time.perf_counter()
        model_input = (img_resized - mean_t) / std_t
        t_stage["3_normalize"] = (time.perf_counter() - t0) * 1000

        # 4. Depth inference
        t0 = time.perf_counter()
        with torch.no_grad():
            depth_raw = model(model_input)[0].float()
        torch.cuda.synchronize()
        t_stage["4_depth_infer"] = (time.perf_counter() - t0) * 1000

        # 5. Depth normalize (quantile)
        t0 = time.perf_counter()
        flat = depth_raw.reshape(-1)
        idx = torch.randint(0, flat.numel(), (16384,), device=flat.device)
        sample = flat[idx]
        q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=flat.device))
        low, high = q_vals[0], q_vals[1]
        depth_norm = ((depth_raw - low) / (high - low)).clamp(0.0, 1.0)
        torch.cuda.synchronize()
        t_stage["5_depth_normalize"] = (time.perf_counter() - t0) * 1000

        # 6. Resize depth to original
        t0 = time.perf_counter()
        near_score = F.interpolate(
            depth_norm[None, None, :, :],
            size=(h, w),
            mode="bilinear", align_corners=False
        )[0, 0]
        disparity = near_score * max_disparity
        torch.cuda.synchronize()
        t_stage["6_resize_depth"] = (time.perf_counter() - t0) * 1000

        # 7. Sharpen edges
        t0 = time.perf_counter()
        disparity_sharp, unreliable = b.sharpen_disparity_edges(
            disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
        )
        torch.cuda.synchronize()
        t_stage["7_sharpen_edges"] = (time.perf_counter() - t0) * 1000

        # 8. GPU warp
        t0 = time.perf_counter()
        left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )
        torch.cuda.synchronize()
        t_stage["8_gpu_warp"] = (time.perf_counter() - t0) * 1000
        hole_counts.append(hole.sum().item())

        # 9. Dilate hole
        t0 = time.perf_counter()
        hole_dilated = dilate_hole_right_adaptive_gpu(hole, 5, max_disparity, device)
        torch.cuda.synchronize()
        t_stage["9_dilate_hole"] = (time.perf_counter() - t0) * 1000

        # A. Inpaint
        t0 = time.perf_counter()
        right_with_hole = right_warped.clone()
        right_with_hole[hole_dilated] = 0.0
        result, hole_final = inpaint_with_kernel(
            right_with_hole, hole_dilated, kernel, k3, max_iter=30
        )
        torch.cuda.synchronize()
        t_stage["A_inpaint_kernel"] = (time.perf_counter() - t0) * 1000
        inpaint_iters.append(30 if hole_final.any() else "early")

        # B. To CPU + BGR
        t0 = time.perf_counter()
        result_np = (result.cpu().numpy() * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        t_stage["B_to_cpu_bgr"] = (time.perf_counter() - t0) * 1000

        # C. Write video (模拟)
        t0 = time.perf_counter()
        # out_writer.write(result_bgr)  # 跳过实际编码
        t_stage["C_write_video"] = (time.perf_counter() - t0) * 1000

        # 记录
        for k, v in t_stage.items():
            stage_times[k].append(v)

        total_ms = (time.perf_counter() - t_frame_start) * 1000
        frame_times.append(total_ms)

        if (frame_idx + 1) % 20 == 0:
            print(f"{frame_idx+1:5d} {total_ms:8.1f} "
                  f"{t_stage['4_depth_infer']:8.1f} "
                  f"{t_stage['8_gpu_warp']:8.1f} "
                  f"{t_stage['A_inpaint_kernel']:8.1f} "
                  f"{hole_counts[-1]:8,d}")

    cap.release()

    print("\n" + "=" * 80)
    print("【v47 全过程耗时详细分析】")
    print("=" * 80)

    print(f"\n总帧数: {len(frame_times)}")
    print(f"平均每帧总耗时: {np.mean(frame_times):.1f} ms")
    print(f"中间 80% 均值: {np.mean(sorted(frame_times)[len(frame_times)//10:-len(frame_times)//10]):.1f} ms")
    print(f"FPS: {1000 / np.mean(frame_times):.1f}")

    print(f"\n【各阶段平均耗时 (ms)】")
    print("-" * 50)
    total_avg = 0
    for name, times in sorted(stage_times.items()):
        if len(times) > 0:
            avg = np.mean(times)
            total_avg += avg
            pct = avg / np.mean(frame_times) * 100
            print(f"  {name:25s}: {avg:6.1f} ms ({pct:4.1f}%)")

    print("-" * 50)
    print(f"  {'各阶段合计':25s}: {total_avg:6.1f} ms")

    print(f"\n【关键阶段占比】")
    depth_total = (np.mean(stage_times["4_depth_infer"]) +
                   np.mean(stage_times["5_depth_normalize"]) +
                   np.mean(stage_times["6_resize_depth"]))
    print(f"  深度推理 (含后处理): {depth_total:.1f} ms ({depth_total/np.mean(frame_times)*100:.1f}%)")

    warp_total = (np.mean(stage_times["7_sharpen_edges"]) +
                  np.mean(stage_times["8_gpu_warp"]))
    print(f"  DIBR warp (含锐化): {warp_total:.1f} ms ({warp_total/np.mean(frame_times)*100:.1f}%)")

    inpaint_total = (np.mean(stage_times["9_dilate_hole"]) +
                     np.mean(stage_times["A_inpaint_kernel"]))
    print(f"  空洞填充 (含膨胀): {inpaint_total:.1f} ms ({inpaint_total/np.mean(frame_times)*100:.1f}%)")

    other = np.mean(frame_times) - depth_total - warp_total - inpaint_total
    print(f"  其他 (CPU/IO): {other:.1f} ms ({other/np.mean(frame_times)*100:.1f}%)")

    print(f"\n【其他统计】")
    print(f"  平均空洞数: {np.mean(hole_counts):.0f} 像素")
    print(f"  空洞膨胀比例: {np.mean(hole_counts) / (w*h) * 100:.2f}%")

    print("\n" + "=" * 80)
    print("【瓶颈分析结论】")
    print("=" * 80)


if __name__ == "__main__":
    main()
