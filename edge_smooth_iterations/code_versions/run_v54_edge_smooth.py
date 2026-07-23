"""
v54: 前景边缘平滑（消除锯齿）

在 v53 填充完成后，对填充区域与原始前景的交界进行抗锯齿平滑。
只影响填充的空洞区域，不修改原始像素。
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v54_edge_smooth")

KERNEL_SIZE = 15
SMOOTH_RADIUS = 3  # 边缘平滑半径


def create_right_to_left_kernel(kernel_size, device):
    """从右向左填充（默认，用背景）"""
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    kernel[0, 0, :, :half + 1] = 1.0
    return kernel


def create_left_to_right_kernel(kernel_size, device):
    """从左向右填充（备用）"""
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    kernel[0, 0, :, half:] = 1.0
    return kernel


@torch.no_grad()
def dilate_right_gpu(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    """只向右膨胀"""
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
def check_right_has_background(hole, kernel_size):
    h, w = hole.shape
    half = kernel_size // 2
    non_hole = ~hole
    right_has_bg = torch.zeros_like(hole)

    for shift in range(1, half + 1):
        shifted = torch.zeros_like(non_hole)
        shifted[:, :-shift] = non_hole[:, shift:]
        right_has_bg = right_has_bg | shifted

    use_rtl_mask = hole & right_has_bg
    use_ltr_mask = hole & ~right_has_bg
    return use_rtl_mask, use_ltr_mask


@torch.no_grad()
def inpaint_unidirectional(img, hole, kernel_rtl, kernel_ltr, max_iter=12):
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_rtl.shape[-1] // 2
    kernel_size = kernel_rtl.shape[-1]

    k3_rtl = kernel_rtl.repeat(3, 1, 1, 1)
    k3_ltr = kernel_ltr.repeat(3, 1, 1, 1)

    for it in range(max_iter):
        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

        rgb_sum_rtl = F.conv2d(img_nchw, k3_rtl, padding=pad, groups=3)
        weight_sum_rtl = F.conv2d(weight_nchw, kernel_rtl, padding=pad)
        rgb_sum_ltr = F.conv2d(img_nchw, k3_ltr, padding=pad, groups=3)
        weight_sum_ltr = F.conv2d(weight_nchw, kernel_ltr, padding=pad)

        avg_rtl = rgb_sum_rtl / weight_sum_rtl.clamp_min(1e-6)
        avg_rtl_hwc = avg_rtl[0].permute(1, 2, 0)
        avg_ltr = rgb_sum_ltr / weight_sum_ltr.clamp_min(1e-6)
        avg_ltr_hwc = avg_ltr[0].permute(1, 2, 0)

        use_rtl_mask, use_ltr_mask = check_right_has_background(hole_cur, kernel_size)
        can_fill_rtl = use_rtl_mask & (weight_sum_rtl[0, 0] > 0.5)
        can_fill_ltr = use_ltr_mask & (weight_sum_ltr[0, 0] > 0.5)

        if can_fill_rtl.sum() + can_fill_ltr.sum() == 0:
            break

        result[can_fill_rtl] = avg_rtl_hwc[can_fill_rtl]
        result[can_fill_ltr] = avg_ltr_hwc[can_fill_ltr]
        hole_cur[can_fill_rtl] = False
        hole_cur[can_fill_ltr] = False

        if not hole_cur.any():
            break

    return result, hole_cur


@torch.no_grad()
def smooth_filled_edges(img, original_hole, filled_hole, smooth_radius=3, device='cuda'):
    """
    对填充区域的边缘进行平滑（消除锯齿）

    只处理：填充了的空洞像素（即白色毛刺区域），不对原始前景边缘进行平滑
    """
    h, w, c = img.shape
    result = img.clone()

    # 只对填充的区域进行平滑（= 膨胀新增的部分，即白色毛刺）
    filled_region = filled_hole & ~original_hole
    if not filled_region.any():
        return result

    # 创建平滑核
    sigma = smooth_radius / 2
    kernel_size = smooth_radius * 2 + 1
    x = torch.arange(kernel_size, device=device) - smooth_radius
    gaussian = torch.exp(-x**2 / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    gaussian_kernel = gaussian.view(1, 1, 1, kernel_size) * gaussian.view(1, 1, kernel_size, 1)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel_3c = gaussian_kernel.repeat(3, 1, 1, 1)

    # 对整图进行平滑
    img_nchw = img.permute(2, 0, 1).unsqueeze(0)
    smoothed = F.conv2d(img_nchw, gaussian_kernel_3c, padding=smooth_radius, groups=3)[0].permute(1, 2, 0)

    # 只把填充区域替换成平滑后的
    result[filled_region] = smoothed[filled_region]

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v54 边缘平滑] 设备: {device}")
    print(f"[v54 边缘平滑] 卷积核: {KERNEL_SIZE}, 平滑半径: {SMOOTH_RADIUS}")
    print(f"[v54 边缘平滑] 策略: v53填充 + 填充边缘抗锯齿")
    print(f"[v54 边缘平滑] 输出目录: {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    kernel_rtl = create_right_to_left_kernel(KERNEL_SIZE, device)
    kernel_ltr = create_left_to_right_kernel(KERNEL_SIZE, device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    h_orig, w_orig = frame_bgr.shape[:2]
    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

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

    hole_dilated = dilate_right_gpu(right_warped, hole, 8, 0.15, device)
    print(f"原始空洞: {hole.sum().item():,}, 膨胀后: {hole_dilated.sum().item():,}")

    # 单帧测试
    print("\n单帧测试...")
    t0 = time.time()
    result, hole_final = inpaint_unidirectional(
        right_warped, hole_dilated, kernel_rtl, kernel_ltr, 12
    )
    t_inpaint = time.time() - t0

    # 边缘平滑
    t0 = time.time()
    result_smoothed = smooth_filled_edges(
        result, hole, hole_dilated, SMOOTH_RADIUS, device
    )
    t_smooth = time.time() - t0

    print(f"  填充耗时: {t_inpaint*1000:.1f} ms")
    print(f"  平滑耗时: {t_smooth*1000:.1f} ms")
    print(f"  剩余空洞: {hole_final.sum().item()}")

    result_np = (result_smoothed.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "single_frame.png"), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

    for crop_name, (y1, y2, x1, x2) in [
        ("face", (200, 500, 600, 900)),
        ("shoulder", (550, 750, 750, 1050)),
    ]:
        crop = result_np[y1:y2, x1:x2]
        cv2.imwrite(str(OUT_DIR / f"crop_{crop_name}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print(f"\n单帧测试完成！开始跑完整视频...")

    # ========== 完整视频 ==========
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {fps:.1f} fps, {total_frames} 帧, {w_orig}×{h_orig}")

    out_path = OUT_DIR / f"v54_smooth_k{KERNEL_SIZE}_r{SMOOTH_RADIUS}.mp4"
    sbs_out_path = OUT_DIR / f"v54_smooth_k{KERNEL_SIZE}_r{SMOOTH_RADIUS}_sbs.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))
    sbs_writer = cv2.VideoWriter(str(sbs_out_path), fourcc, fps, (w_orig * 2, h_orig))

    frame_times = []
    remaining_final = []
    inpaint_times = []
    smooth_times = []

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

        hole_dilated = dilate_right_gpu(right_warped, hole, 8, 0.15, device)

        t_inpaint = time.time()
        result, hole_final = inpaint_unidirectional(
            right_warped, hole_dilated, kernel_rtl, kernel_ltr, 12
        )
        inpaint_times.append((time.time() - t_inpaint) * 1000)

        # 边缘平滑
        t_smooth = time.time()
        result_smoothed = smooth_filled_edges(
            result, hole, hole_dilated, SMOOTH_RADIUS, device
        )
        smooth_times.append((time.time() - t_smooth) * 1000)

        remaining_final.append(hole_final.sum().item())

        result_np = (result_smoothed.cpu().numpy() * 255).astype(np.uint8)
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
            "rem": f"{remaining_final[-1]}"
        })

    cap.release()
    out_writer.release()
    sbs_writer.release()

    print(f"\n[v54 边缘平滑] ✅ 完成！")
    print(f"  共处理 {len(frame_times)} 帧")
    print(f"  平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    print(f"  平均剩余空洞: {np.mean(remaining_final):.1f} 像素")
    print(f"\n  阶段耗时统计:")
    print(f"    填充: {np.mean(inpaint_times):.1f} ms")
    print(f"    边缘平滑: {np.mean(smooth_times):.1f} ms")
    print(f"\n  输出视频: {out_path}")
    print(f"  SBS 对比: {sbs_out_path}")


if __name__ == "__main__":
    main()
