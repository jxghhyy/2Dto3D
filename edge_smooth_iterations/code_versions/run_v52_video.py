"""
v52 完整视频测试：智能混合填充
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
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v52_video_final")

KERNEL_SIZE = 15
LEFT_WIDTH = 5
L2R_ITERS = 5  # 前 5 次从左向右，然后双向收尾
MAX_ITERS = 12


def create_left_to_right_kernel(kernel_size, device):
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    kernel[0, 0, :, half:] = 1.0
    return kernel


def create_bidirectional_kernel(kernel_size, device, left_width=5):
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    x_indices = torch.arange(kernel_size, device=device) - half
    horizontal_mask = x_indices >= -left_width
    kernel[0, 0, :, horizontal_mask] = 1.0
    return kernel


@torch.no_grad()
def dilate_right_gpu(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
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
def inpaint_smart(img, hole, kernel_l2r, kernel_bidir, l2r_iters=5, max_total=8):
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_l2r.shape[-1] // 2

    k3_l2r = kernel_l2r.repeat(3, 1, 1, 1)
    k3_bidir = kernel_bidir.repeat(3, 1, 1, 1)

    for it in range(max_total):
        kernel = kernel_l2r if it < l2r_iters else kernel_bidir
        k3 = k3_l2r if it < l2r_iters else k3_bidir

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v52 智能混合填充] 设备: {device}")
    print(f"[v52 智能混合填充] 卷积核: {KERNEL_SIZE}, 左侧宽度: {LEFT_WIDTH}")
    print(f"[v52 智能混合填充] 策略: 前 {L2R_ITERS} 次从左→右，然后双向收尾 (共 {MAX_ITERS} 次)")
    print(f"[v52 智能混合填充] 输出目录: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    kernel_l2r = create_left_to_right_kernel(KERNEL_SIZE, device)
    kernel_bidir = create_bidirectional_kernel(KERNEL_SIZE, device, LEFT_WIDTH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[v52 智能混合填充] 视频: {fps:.1f} fps, {total_frames} 帧, {w_orig}×{h_orig}")

    out_path = OUT_DIR / f"v52_smart_l2r{L2R_ITERS}_k{KERNEL_SIZE}.mp4"
    sbs_out_path = OUT_DIR / f"v52_smart_l2r{L2R_ITERS}_k{KERNEL_SIZE}_sbs.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))
    sbs_writer = cv2.VideoWriter(str(sbs_out_path), fourcc, fps, (w_orig * 2, h_orig))

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

    pbar = tqdm(range(total_frames), desc="处理帧")
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

        t_dilate = time.time()
        hole_dilated = dilate_right_gpu(right_warped, hole, 8, 0.15, device)
        dilate_times.append((time.time() - t_dilate) * 1000)
        dilated_counts.append(hole_dilated.sum().item() - hole.sum().item())

        t_inpaint = time.time()
        result, hole_final = inpaint_smart(
            right_warped, hole_dilated, kernel_l2r, kernel_bidir, L2R_ITERS, MAX_ITERS
        )
        inpaint_times.append((time.time() - t_inpaint) * 1000)

        remaining_final.append(hole_final.sum().item())

        result_np = (result.cpu().numpy() * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        out_writer.write(result_bgr)

        left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
        left_bgr = cv2.cvtColor(left_uint8, cv2.COLOR_RGB2BGR)
        sbs = np.concatenate([left_bgr, result_bgr], axis=1)
        sbs_writer.write(sbs)

        t1 = time.time()
        frame_times.append(t1 - t0)
        pbar.set_postfix({
            "t/frame": f"{np.mean(frame_times[-10:])*1000:.0f}ms",
            "dilate": f"{np.mean(dilate_times[-10:]):.0f}ms",
            "rem": f"{remaining_final[-1]}"
        })

    cap.release()
    out_writer.release()
    sbs_writer.release()

    print(f"\n[v52 智能混合填充] ✅ 完成！")
    print(f"  共处理 {len(frame_times)} 帧")
    print(f"  平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    print(f"  FPS: {1000 / np.mean(frame_times):.1f}")
    print(f"  平均剩余空洞: {np.mean(remaining_final):.1f} 像素")
    print(f"\n  阶段耗时统计:")
    print(f"    膨胀: {np.mean(dilate_times):.1f} ms")
    print(f"    填充: {np.mean(inpaint_times):.1f} ms")
    print(f"\n  膨胀统计:")
    print(f"    平均每帧膨胀像素: {np.mean(dilated_counts):.0f}")

    print(f"\n【与 v51 性能对比】")
    print(f"  v51 平均耗时: 205.2 ms/帧")
    print(f"  v52 平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    speedup = 205.2 / (np.mean(frame_times) * 1000)
    print(f"  速度变化: {'+' if speedup > 1 else ''}{speedup:.2f}x")

    print(f"\n  输出视频: {out_path}")
    print(f"  SBS 对比: {sbs_out_path}")


if __name__ == "__main__":
    main()
