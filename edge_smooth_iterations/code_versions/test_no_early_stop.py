"""
验证：提前终止 break 是否是问题所在
对比 break vs continue
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v49_no_early_stop")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


@torch.no_grad()
def dilate_with_break(right_warped, hole, max_dilate=8, color_threshold=0.08):
    """原版：遇到不相似就 break"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()

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
                color_diff = np.abs(pixel_color - ref_color).mean()
                if color_diff < color_threshold:
                    hole_dilated[y, check_x] = True
                else:
                    break  # 提前终止

    return hole_dilated


@torch.no_grad()
def dilate_with_continue(right_warped, hole, max_dilate=8, color_threshold=0.08):
    """改进版：遇到不相似继续检测下一个，不 break"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()

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
                color_diff = np.abs(pixel_color - ref_color).mean()
                if color_diff < color_threshold:
                    hole_dilated[y, check_x] = True
                # else: 不相似也继续检测，不 break！

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

    y1, y2 = 200, 500
    x1, x2 = 650, 900

    # 测试不同阈值
    thresholds = [0.05, 0.08, 0.10, 0.15]
    for th in thresholds:
        hole_break = dilate_with_break(right_warped, hole, 8, th)
        hole_continue = dilate_with_continue(right_warped, hole, 8, th)

        new_break = (hole_break.astype(int) - hole.cpu().numpy().astype(int)).sum()
        new_continue = (hole_continue.astype(int) - hole.cpu().numpy().astype(int)).sum()

        print(f"阈值 {th:.2f}: break 版新增 {new_break}, continue 版新增 {new_continue}")

        # 可视化差异
        diff = hole_continue.astype(int) - hole_break.astype(int)
        diff_mask = diff > 0  # continue 多膨胀的区域

        vis = (right_warped * 255).byte().cpu().numpy().copy()
        vis[hole_continue] = 0
        vis[diff_mask] = [0, 255, 0]  # 绿色 = continue 多膨胀的区域

        crop = vis[y1:y2, x1:x2]
        cv2.imwrite(str(OUT_DIR / f"th_{int(th*100):02d}_continue.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print(f"\n✅ 结果已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
