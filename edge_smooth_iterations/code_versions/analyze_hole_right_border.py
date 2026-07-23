"""
分析空洞右边界情况：
1. 空洞右边界 != 图像右边界 → 右边有像素（可以从右往左填充）
2. 空洞右边界 == 图像右边界 → 右边没像素（无法从右往左填充）
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # 加载第70秒帧
    video_path = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    h_orig, w_orig = frame_bgr.shape[:2]

    # 深度推理
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

    dibr_h, dibr_w = h_orig, w_orig
    near_score = F.interpolate(
        depth_norm[None, None, :, :],
        size=(dibr_h, dibr_w),
        mode="bilinear", align_corners=False
    )[0, 0]
    max_disparity = 24.0 * dibr_w / w_orig
    disparity = near_score * max_disparity

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    hole_np = hole.cpu().numpy()

    print(f"图像宽度: {w_orig}")
    print(f"空洞总数: {hole_np.sum()} 像素\n")

    # 逐行分析空洞右边界
    right_not_border = 0  # 空洞右边界 != 图像右边界的像素数
    right_is_border = 0   # 空洞右边界 == 图像右边界的像素数

    right_border_xs = []  # 非边界空洞的右边界x坐标

    for y in range(h_orig):
        row = hole_np[y]
        if not row.any():
            continue
        indices = np.where(row)[0]
        right_x = indices[-1]

        if right_x < w_orig - 1:
            right_not_border += len(indices)
            right_border_xs.append(right_x)
        else:
            right_is_border += len(indices)

    print(f"空洞右边界分析:")
    print(f"  右边界 != 图像边界 (右边有像素): {right_not_border:,} 像素 ({right_not_border/hole_np.sum()*100:.1f}%)")
    print(f"  右边界 == 图像边界 (右边无像素): {right_is_border:,} 像素 ({right_is_border/hole_np.sum()*100:.1f}%)")

    if right_border_xs:
        print(f"\n非边界空洞的右边界 x 统计:")
        print(f"  min: {min(right_border_xs)}")
        print(f"  max: {max(right_border_xs)}")
        print(f"  mean: {np.mean(right_border_xs):.1f}")
        print(f"  median: {np.median(right_border_xs):.1f}")

    # 可视化：把"右边还有像素"的空洞标蓝色，"右边无像素"的标红色
    vis = (right_warped * 255).byte().cpu().numpy().copy()

    for y in range(h_orig):
        row = hole_np[y]
        if not row.any():
            continue
        indices = np.where(row)[0]
        right_x = indices[-1]

        if right_x < w_orig - 1:
            # 蓝色 = 右边还有像素，可以填充
            vis[y, indices] = [0, 0, 255]
        else:
            # 红色 = 右边到边界，无法填充
            vis[y, indices] = [255, 0, 0]

    out_dir = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/hole_right_border_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "hole_classification.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"\n✅ 分类图已保存: {out_dir / 'hole_classification.png'}")


if __name__ == "__main__":
    main()
