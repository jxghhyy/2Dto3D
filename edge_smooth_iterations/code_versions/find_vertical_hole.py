"""
找图像正中间的竖长条空洞带，并放大截图
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
    out_dir = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/vertical_hole_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. 统计每列的空洞像素数，找竖长条 ==========
    hole_per_col = hole_np.sum(axis=0)  # 每列有多少空洞像素

    # 找连续空洞多的列（竖长条）
    print(f"每列空洞像素数统计 (只打印 >50 像素的列):")
    for x in range(w_orig):
        if hole_per_col[x] > 50:
            print(f"  x={x}: {hole_per_col[x]} 空洞像素")

    # ========== 2. 可视化竖长条空洞 ==========
    vis = (right_warped * 255).byte().cpu().numpy().copy()

    # 找空洞列（> 100 像素的）
    vertical_cols = np.where(hole_per_col > 100)[0]
    print(f"\n空洞列 (>100像素): x=[{min(vertical_cols)} ~ {max(vertical_cols)}]")
    print(f"共 {len(vertical_cols)} 列")

    # 高亮这些列为红色
    for x in vertical_cols:
        vis[:, x] = [255, 0, 0]

    cv2.imwrite(str(out_dir / "01_vertical_cols_red_full.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # ========== 3. 放大中间区域 (x=800~1100, y=300~800) ==========
    x1, x2 = 800, 1100
    y1, y2 = 300, 800
    crop = vis[y1:y2, x1:x2]
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    crop_bgr_x2 = cv2.resize(crop_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "02_crop_vertical_x2.png"), crop_bgr_x2)

    # ========== 4. 再放大中间区域 (x=930~980, y=300~800) ==========
    x1, x2 = 930, 980
    y1, y2 = 300, 800
    crop = vis[y1:y2, x1:x2]
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    crop_bgr_x3 = cv2.resize(crop_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "03_crop_vertical_x4.png"), crop_bgr_x3)

    print(f"\n✅ 竖长条空洞分析图已保存到: {out_dir}")
    print(f"  01_vertical_cols_red_full.png - 全图，红色高亮竖空洞")
    print(f"  02_crop_vertical_x2.png - 中间区域放大 x2")
    print(f"  03_crop_vertical_x4.png - 中间区域放大 x4")


if __name__ == "__main__":
    main()
