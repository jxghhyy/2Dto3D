"""
v53 纯单向填充策略

核心逻辑（用户明确要求）：
1. 默认：所有空洞只执行从右向左的卷积（只用右侧像素填充）
   - 绝对不允许空洞左侧向右卷积 → 防止前景扩增
2. 例外：只有当空洞区域右侧完全没有有效像素时（比如画面最右侧）
   - 才允许用从左向右的卷积（可以用大卷积核）

关键：每个像素级判断，不是分阶段！
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v53_pure_unidir")

KERNEL_SIZE = 15


def create_left_to_right_kernel(kernel_size, device):
    """
    从左向右填充的卷积核（默认）
    中心像素只能看到自己的左侧
    左侧 = 背景，防止前景扩增
    """
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    kernel[0, 0, :, half:] = 1.0
    return kernel


def create_right_to_left_kernel(kernel_size, device):
    """
    从右向左填充的卷积核（备用）
    中心像素只能看到自己的右侧
    用于：空洞左侧完全没有背景像素时（最右侧的空洞）
    """
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    kernel[0, 0, :, :half + 1] = 1.0
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
def check_left_has_pixels(hole, kernel_size):
    """
    对每个空洞像素，判断其左侧（卷积核范围内）是否有有效像素
    即：向左找，half 距离内是否有非空洞像素（背景）

    返回:
    - use_ltr_mask: True = 左侧有背景像素，可以用从左向右卷积（默认）
    - use_rtl_mask: True = 左侧没背景像素，只能用从右向左卷积（例外）
    """
    h, w = hole.shape
    half = kernel_size // 2
    device = hole.device

    # 非空洞掩码
    non_hole = ~hole

    # 向左膨胀：非空洞像素向右扩散 half 距离
    # 标记所有"左侧 half 内有非空洞"的位置
    left_has_pixel = torch.zeros_like(hole)

    for shift in range(1, half + 1):
        # 把非空洞掩码向右移 shift 像素（相当于影响右侧的空洞）
        shifted = torch.zeros_like(non_hole)
        shifted[:, shift:] = non_hole[:, :-shift]
        left_has_pixel = left_has_pixel | shifted

    use_ltr_mask = hole & left_has_pixel
    use_rtl_mask = hole & ~left_has_pixel

    return use_ltr_mask, use_rtl_mask


@torch.no_grad()
def inpaint_unidirectional_smart(img, hole, kernel_rtl, kernel_ltr, max_iter=12):
    """
    v53 纯单向智能填充：
    - 右侧有像素 → 从右向左卷积（绝对默认）
    - 右侧没像素 → 从左向右卷积（例外）
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

        # ========== 同时做两种卷积 ==========
        rgb_sum_rtl = F.conv2d(img_nchw, k3_rtl, padding=pad, groups=3)
        weight_sum_rtl = F.conv2d(weight_nchw, kernel_rtl, padding=pad)

        rgb_sum_ltr = F.conv2d(img_nchw, k3_ltr, padding=pad, groups=3)
        weight_sum_ltr = F.conv2d(weight_nchw, kernel_ltr, padding=pad)

        avg_rtl = rgb_sum_rtl / weight_sum_rtl.clamp_min(1e-6)
        avg_rtl_hwc = avg_rtl[0].permute(1, 2, 0)

        avg_ltr = rgb_sum_ltr / weight_sum_ltr.clamp_min(1e-6)
        avg_ltr_hwc = avg_ltr[0].permute(1, 2, 0)

        # ========== 判断每个像素用哪种结果 ==========
        use_ltr_mask, use_rtl_mask = check_left_has_pixels(hole_cur, kernel_size)

        # 能填充的 = 是空洞 且 对应方向权重>0.5
        can_fill_ltr = use_ltr_mask & (weight_sum_ltr[0, 0] > 0.5)
        can_fill_rtl = use_rtl_mask & (weight_sum_rtl[0, 0] > 0.5)

        count_ltr = can_fill_ltr.sum().item()
        count_rtl = can_fill_rtl.sum().item()

        if count_ltr + count_rtl == 0:
            break

        # 合并填充
        result[can_fill_ltr] = avg_ltr_hwc[can_fill_ltr]
        result[can_fill_rtl] = avg_rtl_hwc[can_fill_rtl]

        hole_cur[can_fill_ltr] = False
        hole_cur[can_fill_rtl] = False

        fill_counts.append(count_ltr + count_rtl)
        ltr_counts.append(count_ltr)
        rtl_counts.append(count_rtl)

        if not hole_cur.any():
            break

    return result, hole_cur, fill_counts, rtl_counts, ltr_counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v53 纯单向智能填充] 设备: {device}")
    print(f"[v53 纯单向智能填充] 卷积核: {KERNEL_SIZE}")
    print(f"[v53 纯单向智能填充] 策略: 默认从右向左，右侧无像素时从左向右")
    print(f"[v53 纯单向智能填充] 输出目录: {OUT_DIR}")

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
    result, hole_final, fill_counts, rtl_counts, ltr_counts = inpaint_unidirectional_smart(
        right_warped, hole_dilated, kernel_rtl, kernel_ltr, 12
    )
    t1 = time.time()

    print(f"  剩余空洞: {hole_final.sum().item()}")
    print(f"  耗时: {(t1-t0)*1000:.1f} ms")
    print(f"  每轮填充: {fill_counts}")
    print(f"  从右向左填充: {rtl_counts}")
    print(f"  从左向右填充: {ltr_counts}")
    print(f"  从左向右占比: {sum(ltr_counts)/(sum(fill_counts)+1e-6)*100:.1f}%")

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
    dilated_counts = []
    total_ltr_ratio = []

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
        result, hole_final, fill_counts, rtl_counts, ltr_counts = inpaint_unidirectional_smart(
            right_warped, hole_dilated, kernel_rtl, kernel_ltr, 12
        )
        inpaint_times.append((time.time() - t_inpaint) * 1000)

        remaining_final.append(hole_final.sum().item())
        total_ltr_ratio.append(sum(ltr_counts)/(sum(fill_counts)+1e-6)*100)

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
            "ltr%": f"{np.mean(total_ltr_ratio[-10:]):.0f}%"
        })

    cap.release()
    out_writer.release()
    sbs_writer.release()

    print(f"\n[v53 纯单向智能填充] ✅ 完成！")
    print(f"  共处理 {len(frame_times)} 帧")
    print(f"  平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    print(f"  FPS: {1000 / np.mean(frame_times):.1f}")
    print(f"  平均剩余空洞: {np.mean(remaining_final):.1f} 像素")
    print(f"\n  阶段耗时统计:")
    print(f"    膨胀: {np.mean(dilate_times):.1f} ms")
    print(f"    填充: {np.mean(inpaint_times):.1f} ms")
    print(f"\n  填充方向统计:")
    print(f"    平均从左向右占比: {np.mean(total_ltr_ratio):.1f}%")
    print(f"\n  输出视频: {out_path}")
    print(f"  SBS 对比: {sbs_out_path}")


if __name__ == "__main__":
    main()
