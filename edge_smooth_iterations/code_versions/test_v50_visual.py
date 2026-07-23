"""
v50 单帧可视化测试：对比 v47 vs v50 的膨胀效果
输出带 v50_ 前缀
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v50_visual_comparison")  # 带版本前缀
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


@torch.no_grad()
def dilate_v47_only_right(right_warped, hole, max_dilate=5, color_threshold=0.08):
    """v47: 只向右膨胀"""
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

    return hole_dilated


@torch.no_grad()
def dilate_v50_3direction(right_warped, hole, max_dilate=8, color_threshold=0.15):
    """v50: 左+上+右 三方向膨胀"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold * 255

    # 左右方向
    for y in range(h):
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
            ref_x = start_x - 1
            while ref_x >= 0 and hole_np[y, ref_x]:
                ref_x -= 1
            if ref_x < 0:
                continue
            ref_color = right_warped_np[y, ref_x]

            # 向右
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

            # 向左
            for shift in range(1, max_dilate + 1):
                check_x = start_x - shift
                if check_x < 0:
                    break
                if hole_dilated[y, check_x]:
                    continue
                pixel_color = right_warped_np[y, check_x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()
                if color_diff < th:
                    hole_dilated[y, check_x] = True
                else:
                    break

    # 向上方向
    for x in range(w):
        col = hole_np[:, x]
        indices = np.where(col)[0]
        if len(indices) == 0:
            continue
        regions = []
        start_y = indices[0]
        prev_y = indices[0]
        for i in range(1, len(indices)):
            y = indices[i]
            if y > prev_y + 1:
                regions.append((start_y, prev_y))
                start_y = y
            prev_y = y
        regions.append((start_y, prev_y))

        for start_y, end_y in regions:
            ref_y = start_y - 1
            while ref_y >= 0 and hole_np[ref_y, x]:
                ref_y -= 1
            if ref_y < 0:
                continue
            ref_color = right_warped_np[ref_y, x]

            # 向上
            for shift in range(1, max_dilate + 1):
                check_y = start_y - shift
                if check_y < 0:
                    break
                if hole_dilated[check_y, x]:
                    continue
                pixel_color = right_warped_np[check_y, x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()
                if color_diff < th:
                    hole_dilated[check_y, x] = True
                else:
                    break

    return hole_dilated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v50 可视化对比] 设备: {device}")
    print(f"[v50 可视化对比] 输出目录: {OUT_DIR}")
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

    original_count = hole.sum().item()
    print(f"原始空洞: {original_count:,} 像素")

    right_np = (right_warped * 255).byte().cpu().numpy()

    # ========== v47: 只向右 ==========
    hole_v47 = dilate_v47_only_right(right_warped, hole, 5, 0.08)
    count_v47 = hole_v47.sum() - original_count
    print(f"v47 (只向右, 0.08): 新增 {count_v47:,} 像素")

    # ========== v50: 三方向 ==========
    hole_v50 = dilate_v50_3direction(right_warped, hole, 8, 0.15)
    count_v50 = hole_v50.sum() - original_count
    print(f"v50 (三方向, 0.15): 新增 {count_v50:,} 像素")

    # ========== 可视化输出 ==========
    # 1. 原始空洞
    vis_original = right_np.copy()
    vis_original[hole.cpu().numpy()] = 0
    cv2.imwrite(str(OUT_DIR / "v50_01_original_hole.png"), cv2.cvtColor(vis_original, cv2.COLOR_RGB2BGR))

    # 2. v47 只向右
    vis_v47 = right_np.copy()
    vis_v47[hole_v47] = 0
    cv2.imwrite(str(OUT_DIR / "v50_02_v47_only_right.png"), cv2.cvtColor(vis_v47, cv2.COLOR_RGB2BGR))

    # 3. v50 三方向
    vis_v50 = right_np.copy()
    vis_v50[hole_v50] = 0
    cv2.imwrite(str(OUT_DIR / "v50_03_3direction.png"), cv2.cvtColor(vis_v50, cv2.COLOR_RGB2BGR))

    # 4. 差异高亮
    diff = hole_v50.astype(int) - hole_v47.astype(int)
    diff_mask_new = diff > 0  # v50 新增的（绿色高亮）
    print(f"  v50 比 v47 多膨胀了: {diff_mask_new.sum():,} 像素")

    vis_diff = right_np.copy()
    vis_diff[hole_v50] = 0
    vis_diff[diff_mask_new] = [0, 255, 0]  # 绿色 = v50 多消除的白色毛刺/弧线
    cv2.imwrite(str(OUT_DIR / "v50_04_diff_highlight_green.png"), cv2.cvtColor(vis_diff, cv2.COLOR_RGB2BGR))

    # 5. 裁剪对比：脸区域 + 肩膀区域
    for crop_name, (y1, y2, x1, x2) in [
        ("face", (200, 500, 600, 900)),
        ("shoulder", (550, 750, 750, 1050)),
    ]:
        crop_v47 = vis_v47[y1:y2, x1:x2]
        crop_v50 = vis_v50[y1:y2, x1:x2]
        crop_diff = vis_diff[y1:y2, x1:x2]

        cv2.imwrite(str(OUT_DIR / f"v50_crop_{crop_name}_v47.png"), cv2.cvtColor(crop_v47, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"v50_crop_{crop_name}_v50.png"), cv2.cvtColor(crop_v50, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"v50_crop_{crop_name}_diff.png"), cv2.cvtColor(crop_diff, cv2.COLOR_RGB2BGR))

    print(f"\n✅ v50 可视化对比完成！")
    print(f"  输出目录: {OUT_DIR}")
    print(f"  重点看 v50_crop_face_diff.png 和 v50_crop_shoulder_diff.png")
    print(f"  绿色部分 = v50 多消除的白色毛刺/弧线！")


if __name__ == "__main__":
    main()
