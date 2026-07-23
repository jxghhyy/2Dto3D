"""
v53 最终修正版：纯单向填充策略

核心逻辑（用户反复确认）：
- 空洞左侧 = 前景，空洞右侧 = 背景
- 默认：只从右向左填充（用背景色），绝对禁止前景向左扩增
- 例外：只有当空洞右侧完全没有有效像素时（比如画面最右侧），才用左侧像素从左向右填充
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v53_final_correct")

KERNEL_SIZE = 15


def create_right_to_left_kernel(kernel_size, device):
    """
    从右向左填充的卷积核（默认）
    中心像素只能看到自己的右侧（背景）
    防止前景色向左扩增
    """
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    kernel[0, 0, :, :half + 1] = 1.0
    return kernel


def create_left_to_right_kernel(kernel_size, device):
    """
    从左向右填充的卷积核（备用）
    中心像素只能看到自己的左侧（前景）
    只有当空洞右侧完全没有背景像素时才用
    """
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    kernel[0, 0, :, half:] = 1.0
    return kernel


@torch.no_grad()
def dilate_right_gpu(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    """只向右膨胀（v51 修复版）"""
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
    """
    对每个空洞像素，判断其右侧（卷积核范围内）是否有有效背景像素

    返回:
    - use_rtl_mask: True = 右侧有背景，可以用从右向左卷积（默认）
    - use_ltr_mask: True = 右侧没背景，只能用左侧（前景）从左向右卷积（例外）
    """
    h, w = hole.shape
    half = kernel_size // 2

    non_hole = ~hole

    # 向右膨胀：非空洞像素向左扩散 half 距离
    # 标记所有"右侧 half 内有非空洞"的位置
    right_has_bg = torch.zeros_like(hole)

    for shift in range(1, half + 1):
        # 把非空洞掩码向左移 shift 像素（相当于标记其左侧 shift 距离内的位置）
        shifted = torch.zeros_like(non_hole)
        shifted[:, :-shift] = non_hole[:, shift:]
        right_has_bg = right_has_bg | shifted

    use_rtl_mask = hole & right_has_bg
    use_ltr_mask = hole & ~right_has_bg

    return use_rtl_mask, use_ltr_mask


@torch.no_grad()
def inpaint_unidirectional(img, hole, kernel_rtl, kernel_ltr, max_iter=12):
    """
    v53 智能单向填充
    1. 默认从右向左（用背景）
    2. 右侧没像素时才从左向右（用前景）
    """
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_rtl.shape[-1] // 2
    kernel_size = kernel_rtl.shape[-1]

    k3_rtl = kernel_rtl.repeat(3, 1, 1, 1)
    k3_ltr = kernel_ltr.repeat(3, 1, 1, 1)

    fill_counts = []
    rtl_counts = []
    ltr_counts = []

    for it in range(max_iter):
        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

        # ========== 同时计算两种卷积 ==========
        rgb_sum_rtl = F.conv2d(img_nchw, k3_rtl, padding=pad, groups=3)
        weight_sum_rtl = F.conv2d(weight_nchw, kernel_rtl, padding=pad)

        rgb_sum_ltr = F.conv2d(img_nchw, k3_ltr, padding=pad, groups=3)
        weight_sum_ltr = F.conv2d(weight_nchw, kernel_ltr, padding=pad)

        avg_rtl = rgb_sum_rtl / weight_sum_rtl.clamp_min(1e-6)
        avg_rtl_hwc = avg_rtl[0].permute(1, 2, 0)

        avg_ltr = rgb_sum_ltr / weight_sum_ltr.clamp_min(1e-6)
        avg_ltr_hwc = avg_ltr[0].permute(1, 2, 0)

        # ========== 判断每个像素用哪种方向 ==========
        use_rtl_mask, use_ltr_mask = check_right_has_background(hole_cur, kernel_size)

        # 能填充的 = 是空洞 且 对应方向权重>0.5
        can_fill_rtl = use_rtl_mask & (weight_sum_rtl[0, 0] > 0.5)
        can_fill_ltr = use_ltr_mask & (weight_sum_ltr[0, 0] > 0.5)

        count_rtl = can_fill_rtl.sum().item()
        count_ltr = can_fill_ltr.sum().item()

        if count_rtl + count_ltr == 0:
            break

        # 合并填充
        result[can_fill_rtl] = avg_rtl_hwc[can_fill_rtl]
        result[can_fill_ltr] = avg_ltr_hwc[can_fill_ltr]

        hole_cur[can_fill_rtl] = False
        hole_cur[can_fill_ltr] = False

        fill_counts.append(count_rtl + count_ltr)
        rtl_counts.append(count_rtl)
        ltr_counts.append(count_ltr)

        if not hole_cur.any():
            break

    return result, hole_cur, fill_counts, rtl_counts, ltr_counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v53 最终版] 设备: {device}")
    print(f"[v53 最终版] 卷积核: {KERNEL_SIZE}")
    print(f"[v53 最终版] 策略: 默认从右向左（用背景），右侧没像素时才从左向右")
    print(f"[v53 最终版] 输出目录: {OUT_DIR}")

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
    result, hole_final, fill_counts, rtl_counts, ltr_counts = inpaint_unidirectional(
        right_warped, hole_dilated, kernel_rtl, kernel_ltr, 12
    )
    t1 = time.time()

    print(f"  剩余空洞: {hole_final.sum().item()}")
    print(f"  耗时: {(t1-t0)*1000:.1f} ms")
    print(f"  每轮填充: {fill_counts}")
    print(f"  从右向左填充（默认，用背景）: {rtl_counts}")
    print(f"  从左向右填充（例外，用前景）: {ltr_counts}")
    total_fill = sum(fill_counts)
    rtl_ratio = sum(rtl_counts) / total_fill * 100 if total_fill > 0 else 0
    ltr_ratio = sum(ltr_counts) / total_fill * 100 if total_fill > 0 else 0
    print(f"  从右向左占比: {rtl_ratio:.1f}%")
    print(f"  从左向右占比: {ltr_ratio:.1f}%")

    result_np = (result.cpu().numpy() * 255).astype(np.uint8)
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

    out_path = OUT_DIR / f"v53_unidir_k{KERNEL_SIZE}.mp4"
    sbs_out_path = OUT_DIR / f"v53_unidir_k{KERNEL_SIZE}_sbs.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))
    sbs_writer = cv2.VideoWriter(str(sbs_out_path), fourcc, fps, (w_orig * 2, h_orig))

    frame_times = []
    remaining_final = []
    dilate_times = []
    inpaint_times = []
    ltr_ratios = []

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

        t_inpaint = time.time()
        result, hole_final, fill_counts, rtl_counts, ltr_counts = inpaint_unidirectional(
            right_warped, hole_dilated, kernel_rtl, kernel_ltr, 12
        )
        inpaint_times.append((time.time() - t_inpaint) * 1000)

        remaining_final.append(hole_final.sum().item())
        total_fill = sum(fill_counts)
        if total_fill > 0:
            ltr_ratios.append(sum(ltr_counts) / total_fill * 100)

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
            "rem": f"{remaining_final[-1]}",
            "ltr%": f"{np.mean(ltr_ratios[-10:]):.0f}%" if ltr_ratios else "0%"
        })

    cap.release()
    out_writer.release()
    sbs_writer.release()

    print(f"\n[v53 最终版] ✅ 完成！")
    print(f"  共处理 {len(frame_times)} 帧")
    print(f"  平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    print(f"  平均剩余空洞: {np.mean(remaining_final):.1f} 像素")
    print(f"\n  阶段耗时统计:")
    print(f"    膨胀: {np.mean(dilate_times):.1f} ms")
    print(f"    填充: {np.mean(inpaint_times):.1f} ms")
    print(f"\n  填充方向统计:")
    print(f"    平均从左向右占比: {np.mean(ltr_ratios):.1f}% (应该是少数，例外情况)")
    print(f"    平均从右向左占比: {100 - np.mean(ltr_ratios):.1f}% (默认情况)")
    print(f"\n  输出视频: {out_path}")
    print(f"  SBS 对比: {sbs_out_path}")


if __name__ == "__main__":
    main()
