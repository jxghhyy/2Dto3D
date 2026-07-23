"""
测试 v49 不同阈值的膨胀效果
验证：颜色相似度检测是否真的能把右侧的白色前景残留变成空洞
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v49_threshold_test")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


@torch.no_grad()
def dilate_by_color_similarity(right_warped, hole, max_dilate=8, color_threshold=0.08):
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
                    break

    return torch.from_numpy(hole_dilated).to(right_warped.device)


def main():
    import time
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

    # 裁剪区域：肩膀位置
    y1, y2 = 200, 500
    x1, x2 = 650, 900

    # 保存原图裁剪（用于颜色分析）
    right_np = (right_warped * 255).byte().cpu().numpy()
    crop_original = right_np[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "00_original_crop.png"), cv2.cvtColor(crop_original, cv2.COLOR_RGB2BGR))

    # ========== 采样分析颜色 ==========
    print(f"\n【采样分析肩膀区域的颜色】")
    print(f"  空洞左边界的像素（参考颜色） vs 空洞右边界的像素")

    hole_np = hole.cpu().numpy()
    sample_count = 0
    for y in range(y1, y2, 5):
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        # 找到在 x1-x2 范围内的空洞
        for x in indices:
            if x1 <= x <= x2 and x > 0:
                # 检查是否是左边界
                if x > 0 and not hole_np[y, x-1]:
                    ref_pixel = right_np[y, x-1]

                    # 找到这个空洞的右边界
                    end_x = x
                    while end_x < w_orig and hole_np[y, end_x]:
                        end_x += 1
                    end_x -= 1

                    # 右边界右边的像素
                    if end_x + 1 < w_orig:
                        right_pixel = right_np[y, end_x + 1]
                        diff = np.abs(ref_pixel.astype(float) - right_pixel.astype(float)).mean()

                        if sample_count < 15:  # 打印前 15 个采样
                            print(f"    y={y}: 参考({x-1})={ref_pixel}, 右边({end_x+1})={right_pixel}, Δ={diff:.1f}")
                        sample_count += 1

    print(f"\n  共采样 {sample_count} 个边界点")

    # ========== 测试不同阈值 ==========
    thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    print(f"\n【测试不同阈值的膨胀效果】")
    for th in thresholds:
        hole_dilated = dilate_by_color_similarity(right_warped, hole, 8, th)
        new_pixels = hole_dilated.sum().item() - hole.sum().item()
        print(f"  阈值 {th:.2f}: 新增 {new_pixels:4d} 像素")

        # 可视化
        vis = (right_warped * 255).byte().cpu().numpy().copy()
        vis[hole_dilated.cpu().numpy()] = 0
        crop = vis[y1:y2, x1:x2]
        cv2.imwrite(str(OUT_DIR / f"th_{int(th*100):02d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    # ========== v47 的固定膨胀用于对比 ==========
    hole_float = hole.float()
    cumsum_from_right = torch.cumsum(hole_float.flip(dims=[1]), dims=1).flip(dims=[1])
    hole_widths = cumsum_from_right * hole_float
    dilate_per_pixel = (hole_widths / 24.0 * 5).round().long()
    dilate_per_pixel = torch.clamp(dilate_per_pixel, 1, 5)
    hole_v47 = hole.clone()
    for shift in range(1, 5 + 1):
        shifted = torch.roll(hole, shifts=shift, dims=1)
        shifted[:, :shift] = False
        should_dilate = (dilate_per_pixel >= shift) & shifted
        hole_v47 = hole_v47 | should_dilate

    print(f"  v47 自适应: 新增 {hole_v47.sum().item() - hole.sum().item():4d} 像素")

    vis_v47 = (right_warped * 255).byte().cpu().numpy().copy()
    vis_v47[hole_v47.cpu().numpy()] = 0
    crop_v47 = vis_v47[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "v47_reference.png"), cv2.cvtColor(crop_v47, cv2.COLOR_RGB2BGR))

    print(f"\n✅ 所有结果已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
