"""
v55对照实验：v51原版参数（kernel_size=15, max_iter=8）
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

KERNEL_SIZE = 15
LEFT_WIDTH = 5
MAX_ITER = 8
COLOR_THRESHOLD = 0.15
MAX_DILATE = 8
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v55_small_kernel_test")


def create_asymmetric_kernel(kernel_size, device, left_width=5):
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
def dilate_right_gpu_simple(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold * 255

    for y in range(h):
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        if len(indices) == 1:
            regions = [(indices[0], indices[0])]
        else:
            diff = np.diff(indices)
            splits = np.where(diff > 1)[0] + 1
            if len(splits) == 0:
                regions = [(indices[0], indices[-1])]
            else:
                regions = []
                prev = 0
                for s in splits:
                    regions.append((indices[prev], indices[s-1]))
                    prev = s
                regions.append((indices[prev], indices[-1]))

        for start_x, end_x in regions:
            if start_x <= 0:
                continue
            ref_color = right_warped_np[y, start_x - 1]

            for shift in range(1, max_dilate + 1):
                check_x = end_x + shift
                if check_x >= w:
                    break
                if hole_dilated[y, check_x]:
                    continue

                pixel_color = right_warped_np[y, check_x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()

                if color_diff < th:
                    hole_dilated[y, check_x] = True
                else:
                    break

    return torch.from_numpy(hole_dilated).to(device)


@torch.no_grad()
def inpaint_with_kernel_debug(img, hole, kernel, k3, max_iter=8):
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2

    iter_stats = []

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
        filled_count = can_fill.sum().item()
        iter_stats.append(filled_count)

        if filled_count == 0:
            break

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

    return result, hole_cur, iter_stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v51对照] 设备: {device}")
    print(f"[v51对照] 填充: 不对称核 size={KERNEL_SIZE}, left_width={LEFT_WIDTH}, max_iter={MAX_ITER}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    kernel = create_asymmetric_kernel(KERNEL_SIZE, device, LEFT_WIDTH)
    k3 = kernel.repeat(3, 1, 1, 1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    remaining_final = []
    all_iter_stats = []

    pbar = tqdm(range(min(100, total_frames)), desc="处理帧")
    for frame_idx in pbar:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
        img_resized = F.interpolate(img, size=(1064, 1920), mode="bilinear", align_corners=False)
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
            size=(1080, 1920),
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

        hole_dilated = dilate_right_gpu_simple(
            right_warped, hole, MAX_DILATE, COLOR_THRESHOLD, device
        )

        right_with_hole = right_warped.clone()
        right_with_hole[hole_dilated] = 0.0

        result, hole_final, iter_stats = inpaint_with_kernel_debug(
            right_with_hole, hole_dilated, kernel, k3, MAX_ITER
        )

        remaining_final.append(hole_final.sum().item())
        all_iter_stats.append(iter_stats)

        pbar.set_postfix({
            "rem": f"{remaining_final[-1]}",
            "iters": f"{len(iter_stats)}"
        })

    cap.release()

    print(f"\n[v51对照] ✅ 完成！")
    print(f"  共处理 {len(remaining_final)} 帧")
    print(f"  平均剩余空洞: {np.mean(remaining_final):.1f} 像素")
    print(f"  最大剩余空洞: {np.max(remaining_final):.0f} 像素")

    print(f"\n  迭代统计:")
    avg_iters = np.mean([len(s) for s in all_iter_stats])
    print(f"    平均实际迭代次数: {avg_iters:.1f}")
    for i in range(MAX_ITER):
        counts = [s[i] if i < len(s) else 0 for s in all_iter_stats]
        avg_count = np.mean(counts)
        if avg_count > 0:
            print(f"    第{i+1}轮填充: {avg_count:.0f} 像素")

    early_stop_count = sum(1 for s in all_iter_stats if len(s) < MAX_ITER)
    print(f"\n  提前终止的帧数: {early_stop_count}/{len(remaining_final)} ({early_stop_count/len(remaining_final)*100:.1f}%)")


if __name__ == "__main__":
    main()
