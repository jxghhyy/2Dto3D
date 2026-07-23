"""
专门找肩膀右边的那条白色弧线
用户说：肩膀右侧黑洞之后有一个白色弧线
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/find_white_arc")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


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

    right_np = (right_warped * 255).byte().cpu().numpy()
    hole_np = hole.cpu().numpy()

    # ========== 专注肩膀区域：y 从 550 到 750 ==========
    y1, y2 = 550, 750
    x1, x2 = 750, 1050

    crop_original = right_np[y1:y2, x1:x2].copy()
    cv2.imwrite(str(OUT_DIR / "01_shoulder_crop.png"), cv2.cvtColor(crop_original, cv2.COLOR_RGB2BGR))

    # ========== 逐行详细分析：找空洞右边的白色像素 ==========
    print(f"\n【详细分析空洞右边界 (y={y1}~{y2})】")
    print(f"  左边界参考 vs 右边第 1/2/3/4/5 个像素")

    suspicious_count = 0
    suspicious_points = []  # 记录可疑的白色弧线像素

    for y in range(y1, y2, 2):  # 每 2 行采样
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        # 找到连续空洞区域
        regions = []
        start = None
        for i in range(len(indices)):
            if start is None:
                start = indices[i]
                prev = indices[i]
            elif indices[i] > prev + 1:
                regions.append((start, prev))
                start = indices[i]
            prev = indices[i]
        if start is not None:
            regions.append((start, prev))

        for start_x, end_x in regions:
            # 只看在裁剪范围内的空洞
            if not (x1 <= start_x <= x2 and x1 <= end_x <= x2):
                continue

            if start_x <= 0:
                continue

            ref_pixel = right_np[y, start_x - 1]  # 空洞左边的参考颜色

            # 只检测那些"看起来像白色"的参考（排除黑色背景）
            if ref_pixel.mean() < 150:
                continue  # 参考颜色太暗，不是衣服

            print(f"\n  y={y}, 空洞[{start_x}:{end_x}], 参考={ref_pixel}")

            # 向右检测 10 个像素
            for shift in range(1, 11):
                check_x = end_x + shift
                if check_x >= w_orig:
                    break

                pixel = right_np[y, check_x]
                diff = np.abs(ref_pixel.astype(float) - pixel.astype(float)).mean()

                # 颜色差异小，且像素比较亮（是白色不是黑色）
                if diff < 50 and pixel.mean() > 100:
                    print(f"    ✅ shift={shift}: {pixel}, Δ={diff:.1f} <-- 颜色相似！")
                    suspicious_count += 1
                    suspicious_points.append((y, check_x))
                else:
                    print(f"    shift={shift}: {pixel}, Δ={diff:.1f}")

    print(f"\n  共发现 {suspicious_count} 个可疑像素（颜色相似且较亮）")

    # ========== 可视化高亮：把可疑点标记成红色 ==========
    vis_highlight = right_np.copy()
    for (y, x) in suspicious_points:
        vis_highlight[y, x] = [255, 0, 0]  # 红色标记

    crop_highlight = vis_highlight[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "02_suspicious_arc_red.png"), cv2.cvtColor(crop_highlight, cv2.COLOR_RGB2BGR))

    print(f"\n✅ 结果已保存到: {OUT_DIR}")
    print(f"  01_shoulder_crop.png - 原始裁剪")
    print(f"  02_suspicious_arc_red.png - 红色标记可疑的白色弧线")


if __name__ == "__main__":
    main()
