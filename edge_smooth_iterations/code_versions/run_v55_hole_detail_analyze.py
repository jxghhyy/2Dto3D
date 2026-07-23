"""
详细分析最大空洞的逐行宽度分布
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v55_small_kernel_test")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 跳到第70秒
    target_frame = int(70 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    print(f"读取第 {target_frame} 帧（70秒处）")

    ok, frame_bgr = cap.read()
    if not ok:
        print("无法读取视频")
        return

    h, w = frame_bgr.shape[:2]
    print(f"帧尺寸: {w}x{h}")

    # 深度估计
    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    input_size = 518
    scale = input_size / max(h, w)
    depth_h = max(14, int(round(h * scale / 14)) * 14)
    depth_w = max(14, int(round(w * scale / 14)) * 14)
    if w >= h:
        depth_w = input_size
    else:
        depth_h = input_size

    img_resized = F.interpolate(img, size=(depth_h, depth_w), mode="bilinear", align_corners=False)
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
        size=(h, w),
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

    hole_np = hole.cpu().numpy().astype(np.uint8) * 255

    # ========== 连通域分析 ==========
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hole_np, connectivity=8)

    # 找到最大的空洞（面积最大）
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1  # +1 因为跳过背景0

    left = stats[largest_idx, cv2.CC_STAT_LEFT]
    top = stats[largest_idx, cv2.CC_STAT_TOP]
    width = stats[largest_idx, cv2.CC_STAT_WIDTH]
    height = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    area = stats[largest_idx, cv2.CC_STAT_AREA]

    print(f"\n========== 最大空洞详细分析 ==========")
    print(f"外接矩形: left={left}, top={top}, width={width}, height={height}")
    print(f"总面积: {area} 像素")
    print(f"外接矩形面积: {width * height} 像素")
    print(f"填充率: {area / (width * height) * 100:.1f}%")

    # ========== 逐行分析这个最大空洞 ==========
    row_widths = []
    row_lefts = []
    row_rights = []

    for y in range(top, top + height):
        row = labels[y, left:left + width]
        hole_mask = (row == largest_idx)

        if np.any(hole_mask):
            hole_cols = np.where(hole_mask)[0] + left
            min_x = hole_cols[0]
            max_x = hole_cols[-1]
            actual_width = max_x - min_x + 1

            row_widths.append(actual_width)
            row_lefts.append(min_x)
            row_rights.append(max_x)

    print(f"\n逐行宽度统计:")
    print(f"  平均宽度: {np.mean(row_widths):.1f} 像素")
    print(f"  宽度中位数: {np.median(row_widths):.1f} 像素")
    print(f"  最小宽度: {np.min(row_widths)} 像素")
    print(f"  最大宽度: {np.max(row_widths)} 像素")
    print(f"  90%分位宽度: {np.percentile(row_widths, 90):.1f} 像素")

    # 宽度分布
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300]
    hist, bin_edges = np.histogram(row_widths, bins=bins)
    print(f"\n宽度分布（共 {len(row_widths)} 行有洞）:")
    for i in range(len(bin_edges) - 1):
        if hist[i] > 0:
            print(f"  {bin_edges[i]:3d}-{bin_edges[i+1]:3d} 像素: {hist[i]:4d} 行 ({hist[i]/len(row_widths)*100:.1f}%)")

    # ========== 左边界和右边界的位置分析 ==========
    print(f"\n左边界位置统计:")
    print(f"  平均: {np.mean(row_lefts):.1f}")
    print(f"  标准差: {np.std(row_lefts):.1f}")

    print(f"\n右边界位置统计:")
    print(f"  平均: {np.mean(row_rights):.1f}")
    print(f"  标准差: {np.std(row_rights):.1f}")

    # ========== 分析：每行的右边界是否有有效像素 ==========
    print(f"\n========== 右边界有效性分析 ==========")
    # 对于每行的空洞右边界 x_right，检查 x_right + 1 是否在图像内且不是空洞
    right_has_valid = []
    for i, y in enumerate(range(top, top + height)):
        if i >= len(row_rights):
            continue
        x_right = row_rights[i]
        if x_right + 1 < w:
            is_hole = hole_np[y, x_right + 1] > 0
            right_has_valid.append(not is_hole)
        else:
            right_has_valid.append(False)

    valid_count = sum(right_has_valid)
    print(f"右边界外侧有有效像素的行数: {valid_count}/{len(right_has_valid)} ({valid_count/len(right_has_valid)*100:.1f}%)")

    # ========== 可视化：宽度热图 ==========
    width_heatmap = np.zeros((h, w), dtype=np.uint8)
    max_w = np.max(row_widths)
    for i, y in enumerate(range(top, top + height)):
        if i >= len(row_widths):
            continue
        w_pixel = row_widths[i]
        intensity = int(w_pixel / max_w * 255)
        width_heatmap[y, left:left + width] = intensity

    width_color = cv2.applyColorMap(width_heatmap, cv2.COLORMAP_JET)
    # 在原图上叠加
    overlay = cv2.addWeighted(frame_bgr, 0.6, width_color, 0.4, 0)
    cv2.imwrite(str(OUT_DIR / "largest_hole_width_heatmap.png"), overlay)
    print(f"\n宽度热图已保存: largest_hole_width_heatmap.png")

    # ========== 可视化：左/右边界 ==========
    boundary_viz = frame_bgr.copy()
    for i, y in enumerate(range(top, top + height)):
        if i >= len(row_lefts):
            continue
        # 左边界画红点
        cv2.circle(boundary_viz, (row_lefts[i], y), 1, (0, 0, 255), -1)
        # 右边界画绿点
        cv2.circle(boundary_viz, (row_rights[i], y), 1, (0, 255, 0), -1)

    cv2.imwrite(str(OUT_DIR / "largest_hole_boundaries.png"), boundary_viz)
    print(f"边界可视化已保存: largest_hole_boundaries.png")

    # ========== 打印几行样本 ==========
    print(f"\n========== 前20行样本 ==========")
    for i in range(min(20, len(row_widths))):
        y = top + i
        print(f"  y={y:3d}: 左x={row_lefts[i]:4d}, 右x={row_rights[i]:4d}, 宽度={row_widths[i]:3d}像素, 右侧有背景={right_has_valid[i]}")


if __name__ == "__main__":
    main()
