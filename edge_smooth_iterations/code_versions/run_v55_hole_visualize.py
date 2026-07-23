"""
空洞分布可视化：不同颜色标注每个连通的空洞区域
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


def generate_distinct_colors(n):
    """生成n个差异明显的颜色"""
    colors = []
    for i in range(n):
        hue = int(i * 180 / n)  # OpenCV的HSV中H范围是0-180
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append(color)
    return colors


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频帧率: {fps}")

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

    # DPT v2要求尺寸是14的倍数，用和v51相同的缩放逻辑
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
    print(f"总空洞像素: {np.sum(hole_np > 0)}")

    # ========== 1. 找到所有连通的空洞区域 ==========
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hole_np, connectivity=8)
    print(f"\n检测到 {num_labels - 1} 个独立的空洞区域")

    # 按面积排序
    areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景（标签0）
    sorted_indices = np.argsort(-areas) + 1  # 降序排列

    print(f"\nTop 10 空洞区域：")
    for i, idx in enumerate(sorted_indices[:10]):
        area = stats[idx, cv2.CC_STAT_AREA]
        left = stats[idx, cv2.CC_STAT_LEFT]
        top = stats[idx, cv2.CC_STAT_TOP]
        width = stats[idx, cv2.CC_STAT_WIDTH]
        height = stats[idx, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[idx]
        print(f"  区域{i+1}: {area:6d} 像素 | 位置({left:4d},{top:4d}) | 尺寸{width:4d}x{height:4d} | 中心({cx:.0f},{cy:.0f})")

    # ========== 2. 不同颜色标注每个空洞区域 ==========
    colors = generate_distinct_colors(num_labels)

    # 可视化1：彩色标注的空洞图
    hole_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, num_labels):  # 跳过背景
        mask = (labels == i)
        hole_colored[mask] = colors[i]

    # 可视化2：在原图上半透明叠加空洞
    overlay = frame_bgr.copy()
    for i in range(1, num_labels):
        mask = (labels == i)
        overlay[mask] = frame_bgr[mask] * 0.3 + np.array(colors[i]) * 0.7

    # 可视化3：放大最右侧的大空洞区域
    right_hole = hole_colored[:, -100:]
    right_hole_bgr = frame_bgr[:, -100:].copy()
    right_hole_overlay = right_hole_bgr.copy()
    for i in range(1, num_labels):
        mask = (labels[:, -100:] == i)
        right_hole_overlay[mask] = right_hole_bgr[mask] * 0.3 + np.array(colors[i]) * 0.7

    # ========== 3. 逐行分析空洞的宽度分布 ==========
    hole_widths = []
    for y in range(h):
        row = hole_np[y]
        indices = np.where(row > 0)[0]
        if len(indices) == 0:
            continue
        # 找连续段
        if len(indices) == 1:
            hole_widths.append(1)
        else:
            diff = np.diff(indices)
            splits = np.where(diff > 1)[0] + 1
            if len(splits) == 0:
                hole_widths.append(indices[-1] - indices[0] + 1)
            else:
                prev = 0
                for s in splits:
                    hole_widths.append(indices[s-1] - indices[prev] + 1)
                    prev = s
                hole_widths.append(indices[-1] - indices[prev] + 1)

    print(f"\n空洞宽度统计:")
    print(f"  平均宽度: {np.mean(hole_widths):.1f} 像素")
    print(f"  最大宽度: {np.max(hole_widths)} 像素")
    print(f"  宽度中位数: {np.median(hole_widths):.1f} 像素")
    print(f"  90%分位: {np.percentile(hole_widths, 90):.1f} 像素")

    # 宽度分布统计
    bins = [0, 3, 5, 10, 20, 50, 100, 1000]
    hist, _ = np.histogram(hole_widths, bins=bins)
    print(f"\n宽度分布:")
    for i in range(len(bins)-1):
        print(f"  {bins[i]:3d}-{bins[i+1]:3d} 像素: {hist[i]:5d} 个")

    # ========== 4. 保存结果 ==========
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(OUT_DIR / "hole_colored.png"), hole_colored)
    cv2.imwrite(str(OUT_DIR / "hole_overlay.png"), overlay)
    cv2.imwrite(str(OUT_DIR / "right_hole_detail.png"), right_hole_overlay)

    # 拼接大图
    combined = np.hstack([frame_bgr, overlay])
    cv2.imwrite(str(OUT_DIR / "hole_visualization_combined.png"), combined)

    print(f"\n结果已保存到: {OUT_DIR}")
    print(f"  - hole_colored.png: 纯彩色标注的空洞")
    print(f"  - hole_overlay.png: 原图叠加空洞")
    print(f"  - right_hole_detail.png: 最右侧100列放大")
    print(f"  - hole_visualization_combined.png: 原图和标注对比")


if __name__ == "__main__":
    main()
