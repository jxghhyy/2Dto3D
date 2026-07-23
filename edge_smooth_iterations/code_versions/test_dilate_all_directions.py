"""
验证：向所有方向膨胀边界
用户发现的白色毛刺在空洞的左/上边界，不在右边界！
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/dilate_all_directions")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


@torch.no_grad()
def dilate_boundary_based(right_warped, hole, max_dilate=5, color_threshold=0.08):
    """
    向所有方向膨胀空洞边界：
    对每个边界像素，检测四个方向的相邻像素，如果颜色相似就变成空洞
    """
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()

    # 找到所有边界像素（是空洞，但 4-邻域有非空洞）
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    dilated_hole = cv2.dilate(hole_np.astype(np.uint8), kernel)
    boundary = dilated_hole & (~hole_np)  # 膨胀后新增的就是边界附近

    boundary_y, boundary_x = np.where(boundary)

    print(f"  找到 {len(boundary_y)} 个边界像素")

    # 对每个边界像素，向四个方向检测
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上, 下, 左, 右

    for y, x in zip(boundary_y, boundary_x):
        # 找到这个边界像素对应的"外部参考颜色"（空洞外的第一个非空洞像素）
        ref_color = None
        for dy, dx in directions:
            ref_y, ref_x = y + dy, x + dx
            if 0 <= ref_y < h and 0 <= ref_x < w and not hole_np[ref_y, ref_x]:
                ref_color = right_warped_np[ref_y, ref_x]
                break

        if ref_color is None:
            continue

        # 用这个参考颜色，检测边界周围的像素
        for dy, dx in directions:
            for shift in range(1, max_dilate + 1):
                check_y = y + dy * shift
                check_x = x + dx * shift
                if not (0 <= check_y < h and 0 <= check_x < w):
                    break
                if hole_dilated[check_y, check_x]:
                    continue

                pixel_color = right_warped_np[check_y, check_x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()

                if color_diff < color_threshold * 255:
                    hole_dilated[check_y, check_x] = True
                else:
                    break

    return hole_dilated


@torch.no_grad()
def dilate_left_up_specific(right_warped, hole, max_dilate=5, color_threshold=0.08):
    """
    专门针对用户的问题：只向左和向上膨胀
    """
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()

    for y in range(h):
        for x in range(w):
            if hole_np[y, x]:
                continue  # 只检测非空洞像素，看它是不是应该变成空洞

                # 检查这个像素的右边/下边是不是空洞
                # 如果是，说明这个像素在空洞左边/上边
                should_check = False
                ref_color = None

                # 像素右边是空洞 → 这是空洞左边界的左边像素
                if x + 1 < w and hole_np[y, x + 1]:
                    should_check = True
                    # 找更左边的真实前景作为参考
                    ref_x = x - 1
                    while ref_x >= 0 and hole_np[y, ref_x]:
                        ref_x -= 1
                    if ref_x >= 0:
                        ref_color = right_warped_np[y, ref_x]

                # 像素下边是空洞 → 这是空洞上边界的上边像素
                if y + 1 < h and hole_np[y + 1, x]:
                    should_check = True
                    # 找更上边的真实前景作为参考
                    ref_y = y - 1
                    while ref_y >= 0 and hole_np[ref_y, x]:
                        ref_y -= 1
                    if ref_y >= 0 and ref_color is None:
                        ref_color = right_warped_np[ref_y, x]

                if not should_check or ref_color is None:
                    continue

                # 颜色相似度检测
                pixel_color = right_warped_np[y, x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()

                if color_diff < color_threshold * 255:
                    hole_dilated[y, x] = True

    return hole_dilated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

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

    print(f"原始空洞: {hole.sum().item():,} 像素")

    # 裁剪用户说的区域：中上方（脸/头发）
    y1, y2 = 200, 500
    x1, x2 = 600, 900

    # 原始空洞
    right_np = (right_warped * 255).byte().cpu().numpy()
    vis_original = right_np.copy()
    vis_original[hole.cpu().numpy()] = 0
    crop_original = vis_original[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "01_original_hole.png"), cv2.cvtColor(crop_original, cv2.COLOR_RGB2BGR))
    print("  保存原始空洞图")

    # ========== 方案1：只向右膨胀（v49 原版，用于对比） ==========
    hole_np = hole.cpu().numpy()
    hole_right = hole_np.copy()
    for y in range(h_orig):
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue
        regions = []
        start_x = indices[0]
        prev_x = indices[0]
        for i in range(1, len(indices)):
            x = indices[i]
            if x > prev_x + 1:
                regions.append((start_x, prev_x))
                start_x = x
            prev_x = x
        regions.append((start_x, prev_x))

        for start_x, end_x in regions:
            if start_x <= 0:
                continue
            ref_color = right_np[y, start_x - 1]
            for shift in range(1, 6):
                check_x = end_x + shift
                if check_x >= w_orig:
                    break
                if hole_right[y, check_x]:
                    continue
                pixel_color = right_np[y, check_x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()
                if color_diff < 0.08 * 255:
                    hole_right[y, check_x] = True
                else:
                    break

    new_right = hole_right.astype(int).sum() - hole_np.astype(int).sum()
    print(f"  只向右膨胀: 新增 {new_right} 像素")

    vis_right = right_np.copy()
    vis_right[hole_right] = 0
    crop_right = vis_right[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "02_only_right_dilate.png"), cv2.cvtColor(crop_right, cv2.COLOR_RGB2BGR))

    # ========== 方案2：只向左+向上膨胀（针对用户的问题） ==========
    hole_leftup = hole_np.copy()
    for y in range(h_orig):
        for x in range(w_orig):
            if hole_np[y, x]:
                continue

            # 检查：这个像素的右边或下边是不是空洞？
            is_left_of_hole = (x + 1 < w_orig) and hole_np[y, x + 1]
            is_above_hole = (y + 1 < h_orig) and hole_np[y + 1, x]

            if not (is_left_of_hole or is_above_hole):
                continue

            # 找参考颜色：向左或向上找真实前景
            ref_color = None
            if is_left_of_hole:
                ref_x = x - 1
                while ref_x >= 0 and hole_np[y, ref_x]:
                    ref_x -= 1
                if ref_x >= 0:
                    ref_color = right_np[y, ref_x]
            if is_above_hole and ref_color is None:
                ref_y = y - 1
                while ref_y >= 0 and hole_np[ref_y, x]:
                    ref_y -= 1
                if ref_y >= 0:
                    ref_color = right_np[ref_y, x]

            if ref_color is None:
                continue

            pixel_color = right_np[y, x]
            color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()
            if color_diff < 0.08 * 255:
                hole_leftup[y, x] = True

    new_leftup = hole_leftup.astype(int).sum() - hole_np.astype(int).sum()
    print(f"  左+向上膨胀: 新增 {new_leftup} 像素")

    vis_leftup = right_np.copy()
    vis_leftup[hole_leftup] = 0
    crop_leftup = vis_leftup[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "03_left_up_dilate.png"), cv2.cvtColor(crop_leftup, cv2.COLOR_RGB2BGR))

    # ========== 差异高亮 ==========
    diff = hole_leftup.astype(int) - hole_right.astype(int)
    diff_mask = diff > 0  # 左+上膨胀多出来的区域（用户说的白色毛刺）

    vis_diff = right_np.copy()
    vis_diff[hole_leftup] = 0
    vis_diff[diff_mask] = [0, 255, 0]  # 绿色 = 左+上膨胀新增加的
    crop_diff = vis_diff[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "04_diff_highlight_green.png"), cv2.cvtColor(crop_diff, cv2.COLOR_RGB2BGR))

    print(f"\n✅ 结果已保存到: {OUT_DIR}")
    print(f"  请重点对比 02 (只向右) vs 03 (左+向上)")
    print(f"  04 中绿色的部分，就是用户指出的那些白色毛刺！")


if __name__ == "__main__":
    main()
