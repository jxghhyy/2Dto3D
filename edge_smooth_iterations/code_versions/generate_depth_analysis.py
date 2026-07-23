"""
生成深度图和视差图分析
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
    OUTPUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/depth_analysis")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # ========== 生成深度图 ==========
    near_score_np = near_score.cpu().numpy()

    # 深度图（蓝色=近，红色=远）
    depth_vis = cv2.applyColorMap((near_score_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUTPUT_DIR / "01_near_score_jet.png"), depth_vis)

    # 深度图（灰度）
    depth_gray = (near_score_np * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "02_near_score_gray.png"), depth_gray)

    # 视差图（越大越近）
    disp_np = disparity_sharp.cpu().numpy()
    disp_norm = (disp_np / disp_np.max() * 255).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(OUTPUT_DIR / "03_disparity_jet.png"), disp_vis)

    # ========== 关键：画出空洞左边缘和视差关系 ==========
    hole_np = hole.cpu().numpy()

    # 画每条扫描线的空洞左边缘
    edge_vis = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    for y in range(h_orig):
        row = hole_np[y]
        if row.any():
            left_edge_x = np.where(row)[0][0]  # 空洞的左边界x
            right_edge_x = np.where(row)[0][-1]  # 空洞的右边界x

            # 左边缘画绿点
            cv2.circle(edge_vis, (left_edge_x, y), 1, (0, 255, 0), -1)
            # 右边缘画红点
            cv2.circle(edge_vis, (right_edge_x, y), 1, (255, 0, 0), -1)

    cv2.imwrite(str(OUTPUT_DIR / "04_hole_edges.png"), cv2.cvtColor(edge_vis, cv2.COLOR_RGB2BGR))

    # ========== 画一条扫描线的剖面图 ==========
    scan_y = 540  # 中间行
    scan_row = frame_bgr[scan_y, :, ::-1]  # RGB
    disp_row = disp_np[scan_y, :]
    hole_row = hole_np[scan_y, :]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # 上图：图像扫描线
    ax1.imshow(scan_row[np.newaxis, :, :], aspect='auto')
    ax1.set_title(f'Scanline at y={scan_y}')
    ax1.set_yticks([])

    # 标记空洞区域
    hole_xs = np.where(hole_row)[0]
    if len(hole_xs) > 0:
        ax1.axvspan(hole_xs[0], hole_xs[-1], alpha=0.3, color='red', label='hole')
    ax1.legend()

    # 下图：视差曲线
    ax2.plot(disp_row, 'b-', linewidth=1)
    ax2.set_title('Disparity (higher = closer)')
    ax2.set_ylabel('disparity')
    ax2.set_xlabel('x')
    ax2.grid(True, alpha=0.3)

    # 标记空洞区域
    if len(hole_xs) > 0:
        ax2.axvspan(hole_xs[0], hole_xs[-1], alpha=0.3, color='red')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "05_scanline_profile.png"), dpi=150)
    plt.close()

    # ========== 视差直方图 ==========
    plt.figure(figsize=(10, 5))
    plt.hist(disp_np[~hole_np], bins=50, alpha=0.7, label='valid pixels')
    plt.title('Disparity Histogram')
    plt.xlabel('disparity')
    plt.ylabel('count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(str(OUTPUT_DIR / "06_disparity_histogram.png"), dpi=150)
    plt.close()

    print(f"✅ 深度分析图已输出到: {OUTPUT_DIR}")
    print(f"  01_near_score_jet.png - 深度图（彩色）")
    print(f"  02_near_score_gray.png - 深度图（灰度）")
    print(f"  03_disparity_jet.png - 视差图（彩色）")
    print(f"  04_hole_edges.png - 空洞边缘标记")
    print(f"  05_scanline_profile.png - 扫描线剖面图")
    print(f"  06_disparity_histogram.png - 视差直方图")


if __name__ == "__main__":
    main()
