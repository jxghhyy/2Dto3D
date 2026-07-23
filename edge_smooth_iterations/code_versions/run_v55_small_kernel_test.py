"""
v55: 小卷积核尺寸限制测试

测试内容：
1. 卷积核尺寸从 15 减小到 3（只看左右各1像素）
2. 限制最大迭代次数为 3
3. 观察对"右侧没有像素"的空洞区域的影响

基于 v51 架构（只向右膨胀 + 不对称核填充）
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

# ========== 参数 ==========
KERNEL_SIZE = 3  # v55: 从 15 减小到 3
LEFT_WIDTH = 1   # v55: 左宽也相应减小（k//2=1）
MAX_ITER = 3     # v55: 从 8 限制到 3
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
    """带调试信息的版本：记录每轮迭代填充了多少像素"""
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2

    iter_stats = []  # 记录每轮填充的像素数

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

        if not hole_cur.any():
            break

    return result, hole_cur, iter_stats


def visualize_remaining_hole(hole_final, original_hole, w, h):
    """可视化剩余空洞：红色=原始空洞，蓝色=剩余未填充"""
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[original_hole.cpu().numpy()] = [64, 64, 255]  # 红色：原始空洞
    vis[hole_final.cpu().numpy()] = [255, 64, 64]     # 蓝色：剩余未填充
    return vis


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v55 小卷积核测试] 设备: {device}")
    print(f"[v55 小卷积核测试] 只向右膨胀: 阈值={COLOR_THRESHOLD}, 最大={MAX_DILATE}像素")
    print(f"[v55 小卷积核测试] 填充: 不对称核 size={KERNEL_SIZE}, left_width={LEFT_WIDTH}, max_iter={MAX_ITER}")
    print(f"[v55 小卷积核测试] 输出目录: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    kernel = create_asymmetric_kernel(KERNEL_SIZE, device, LEFT_WIDTH)
    k3 = kernel.repeat(3, 1, 1, 1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[v55 小卷积核测试] 视频: {fps:.1f} fps, {total_frames} 帧, {w_orig}×{h_orig}")

    out_path = OUT_DIR / f"v55_k{KERNEL_SIZE}_iter{MAX_ITER}.mp4"
    debug_out_path = OUT_DIR / f"v55_k{KERNEL_SIZE}_iter{MAX_ITER}_debug.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))
    debug_writer = cv2.VideoWriter(str(debug_out_path), fourcc, fps, (w_orig * 2, h_orig))

    input_size = 518
    scale = input_size / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = input_size
    else:
        depth_h = input_size

    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    frame_times = []
    remaining_final = []
    dilate_times = []
    inpaint_times = []
    dilated_counts = []
    all_iter_stats = []

    pbar = tqdm(range(min(100, total_frames)), desc="处理帧")  # 先测100帧
    for frame_idx in pbar:
        t0 = time.time()

        ok, frame_bgr = cap.read()
        if not ok:
            break

        left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
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
            size=(h_orig, w_orig),
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

        # 只向右膨胀
        t_dilate = time.time()
        hole_dilated = dilate_right_gpu_simple(
            right_warped, hole, MAX_DILATE, COLOR_THRESHOLD, device
        )
        dilate_times.append((time.time() - t_dilate) * 1000)
        dilated_counts.append(hole_dilated.sum().item() - hole.sum().item())

        right_with_hole = right_warped.clone()
        right_with_hole[hole_dilated] = 0.0

        # 小卷积核填充（带调试信息）
        t_inpaint = time.time()
        result, hole_final, iter_stats = inpaint_with_kernel_debug(
            right_with_hole, hole_dilated, kernel, k3, MAX_ITER
        )
        inpaint_times.append((time.time() - t_inpaint) * 1000)

        remaining_final.append(hole_final.sum().item())
        all_iter_stats.append(iter_stats)

        result_np = (result.cpu().numpy() * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        out_writer.write(result_bgr)

        # 调试可视化：左侧结果，右侧剩余空洞
        remaining_vis = visualize_remaining_hole(hole_final, hole_dilated, w_orig, h_orig)
        debug_sbs = np.concatenate([result_bgr, remaining_vis], axis=1)
        debug_writer.write(debug_sbs)

        t1 = time.time()
        frame_times.append(t1 - t0)
        pbar.set_postfix({
            "t/frame": f"{np.mean(frame_times[-10:])*1000:.0f}ms",
            "rem": f"{remaining_final[-1]}",
            "iters": f"{len(iter_stats)}"
        })

    cap.release()
    out_writer.release()
    debug_writer.release()

    print(f"\n[v55 小卷积核测试] ✅ 完成！")
    print(f"  共处理 {len(frame_times)} 帧")
    print(f"  平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    print(f"  FPS: {1000 / np.mean(frame_times):.1f}")
    print(f"  平均剩余空洞: {np.mean(remaining_final):.1f} 像素")
    print(f"  最大剩余空洞: {np.max(remaining_final):.0f} 像素")
    print(f"\n  阶段耗时统计:")
    print(f"    膨胀: {np.mean(dilate_times):.1f} ms")
    print(f"    填充: {np.mean(inpaint_times):.1f} ms")
    print(f"\n  膨胀统计:")
    print(f"    平均每帧膨胀像素: {np.mean(dilated_counts):.0f}")

    # 迭代统计
    print(f"\n  迭代统计:")
    avg_iters = np.mean([len(s) for s in all_iter_stats])
    print(f"    平均实际迭代次数: {avg_iters:.1f}")
    for i in range(MAX_ITER):
        counts = [s[i] if i < len(s) else 0 for s in all_iter_stats]
        avg_count = np.mean(counts)
        print(f"    第{i+1}轮填充: {avg_count:.0f} 像素")

    # 有多少帧因为"没有可填充像素"而提前终止
    early_stop_count = sum(1 for s in all_iter_stats if len(s) < MAX_ITER)
    print(f"\n  提前终止的帧数: {early_stop_count}/{len(frame_times)} ({early_stop_count/len(frame_times)*100:.1f}%)")
    print(f"  （原因：卷积核覆盖不到任何有效像素，无法继续填充）")

    print(f"\n  输出视频: {out_path}")
    print(f"  调试视频: {debug_out_path}")


if __name__ == "__main__":
    main()
