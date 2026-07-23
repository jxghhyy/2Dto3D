"""
调试：分别统计左、上、右三个方向各膨胀了多少像素
分析为什么左边扩大了
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/debug_direction_analysis")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


@torch.no_grad()
def analyze_direction_contribution(right_warped, hole, max_dilate=8, color_threshold=0.15):
    """分别统计三个方向各自的膨胀量"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold * 255

    original_count = hole_np.sum()

    # ========== 只向右 ==========
    hole_right = hole_np.copy()
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
                if hole_right[y, check_x]:
                    continue
                pixel_color = right_warped_np[y, check_x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()
                if color_diff < th:
                    hole_right[y, check_x] = True
                else:
                    break

    count_right = hole_right.sum() - original_count

    # ========== 只向左 ==========
    hole_left = hole_np.copy()
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
                check_x = start_x - shift
                if check_x < 0:
                    break
                if hole_left[y, check_x]:
                    continue
                pixel_color = right_warped_np[y, check_x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()
                if color_diff < th:
                    hole_left[y, check_x] = True
                else:
                    break

    count_left = hole_left.sum() - original_count

    # ========== 只向上 ==========
    hole_up = hole_np.copy()
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
            if start_y <= 0:
                continue
            ref_color = right_warped_np[start_y - 1, x]
            for shift in range(1, max_dilate + 1):
                check_y = start_y - shift
                if check_y < 0:
                    break
                if hole_up[check_y, x]:
                    continue
                pixel_color = right_warped_np[check_y, x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()
                if color_diff < th:
                    hole_up[check_y, x] = True
                else:
                    break

    count_up = hole_up.sum() - original_count

    # ========== 三方向叠加（去重） ==========
    hole_all = hole_np.copy()
    hole_all = hole_all | hole_right | hole_left | hole_up
    count_all = hole_all.sum() - original_count

    print(f"\n【膨胀方向分解统计】")
    print(f"  原始空洞: {original_count:,} 像素")
    print(f"  只向右: {count_right:,} 像素 ({count_right/original_count*100:.1f}%)")
    print(f"  只向左: {count_left:,} 像素 ({count_left/original_count*100:.1f}%)")
    print(f"  只向上: {count_up:,} 像素 ({count_up/original_count*100:.1f}%)")
    print(f"  三方向(去重): {count_all:,} 像素 ({count_all/original_count*100:.1f}%)")

    # 重叠分析
    overlap_left_right = (hole_left & hole_right).sum() - original_count
    print(f"\n  左&右重叠: {max(0, overlap_left_right):,} 像素")

    # ========== 可视化 ==========
    right_np = (right_warped * 255).byte().cpu().numpy()

    # 不同方向用不同颜色标记
    vis = right_np.copy()
    vis[hole_np] = 0  # 原始空洞：黑色
    vis[hole_right & ~hole_np] = [0, 0, 255]  # 蓝色 = 向右新增
    vis[hole_left & ~hole_np & ~hole_right] = [255, 0, 0]  # 红色 = 向左新增
    vis[hole_up & ~hole_np & ~hole_right & ~hole_left] = [0, 255, 0]  # 绿色 = 向上新增

    # 裁剪两个关键区域
    for crop_name, (y1, y2, x1, x2) in [
        ("face", (200, 500, 600, 900)),
        ("shoulder", (550, 750, 750, 1050)),
    ]:
        crop = vis[y1:y2, x1:x2]
        cv2.imwrite(str(OUT_DIR / f"direction_{crop_name}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print(f"\n  图例: 黑色=原始空洞, 蓝色=向右膨胀, 红色=向左膨胀, 绿色=向上膨胀")
    print(f"  裁剪图已保存到: {OUT_DIR}")

    return count_left, count_right, count_up


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

    analyze_direction_contribution(right_warped, hole, 8, 0.15)

    print(f"\n✅ 方向分析完成！")


if __name__ == "__main__":
    main()
