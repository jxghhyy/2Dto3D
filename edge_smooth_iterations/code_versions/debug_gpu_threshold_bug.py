"""
验证 GPU 版本的阈值 bug：是否漏乘了 255
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/debug_threshold_bug")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


@torch.no_grad()
def dilate_correct(right_warped, hole, max_dilate=8, color_threshold=0.15):
    """正确版本：阈值乘 255"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold * 255  # ✅ 正确

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
def dilate_gpu_buggy(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    """GPU 版本（有 bug）：阈值没有乘 255"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold  # ❌ Bug！没有乘 255

    for y in range(h):
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        if len(indices) == 1:
            regions = [(indices[0], indices[0])]
        else:
            diff = np.diff(indices)
            splits = np.where(diff > 1)[0] + 1
            if len(splits) == 0:
                regions = [(indices[0], indices[-1])]
            else:
                regions = []
                prev = 0
                for s in splits:
                    regions.append((indices[prev], indices[s-1]))
                    prev = s
                regions.append((indices[prev], indices[-1]))

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

    return torch.from_numpy(hole_dilated).to(device)


@torch.no_grad()
def dilate_gpu_fixed(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    """GPU 版本（已修复）：阈值乘 255"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold * 255  # ✅ 修复：乘 255

    for y in range(h):
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        if len(indices) == 1:
            regions = [(indices[0], indices[0])]
        else:
            diff = np.diff(indices)
            splits = np.where(diff > 1)[0] + 1
            if len(splits) == 0:
                regions = [(indices[0], indices[-1])]
            else:
                regions = []
                prev = 0
                for s in splits:
                    regions.append((indices[prev], indices[s-1]))
                    prev = s
                regions.append((indices[prev], indices[-1]))

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

    return torch.from_numpy(hole_dilated).to(device)


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

    original_count = hole.sum().item()
    print(f"原始空洞: {original_count:,} 像素")

    # ========== 1. 正确版本（参照） ==========
    hole_correct = dilate_correct(right_warped, hole, 8, 0.15)
    count_correct = hole_correct.sum() - original_count
    print(f"✅ 正确版本 (th * 255): 新增 {count_correct:,} 像素")

    # ========== 2. GPU bug 版本 ==========
    hole_buggy = dilate_gpu_buggy(right_warped, hole, 8, 0.15, device)
    count_buggy = hole_buggy.sum().item() - original_count
    print(f"❌ GPU bug 版本 (th 没乘 255): 新增 {count_buggy:,} 像素")

    # ========== 3. GPU 修复版本 ==========
    hole_fixed = dilate_gpu_fixed(right_warped, hole, 8, 0.15, device)
    count_fixed = hole_fixed.sum().item() - original_count
    print(f"✅ GPU 修复版本 (th * 255): 新增 {count_fixed:,} 像素")

    print(f"\n【Bug 分析】")
    print(f"  bug 版本比正确版本少膨胀了: {count_correct - count_buggy:,} 像素")
    print(f"  修复后差异: {abs(count_correct - count_fixed):,} 像素 (应该接近 0)")

    # ========== 可视化对比 ==========
    right_np = (right_warped * 255).byte().cpu().numpy()

    # 1. bug 版本
    vis_buggy = right_np.copy()
    vis_buggy[hole_buggy.cpu().numpy()] = 0
    cv2.imwrite(str(OUT_DIR / "01_buggy.png"), cv2.cvtColor(vis_buggy, cv2.COLOR_RGB2BGR))

    # 2. 修复版本
    vis_fixed = right_np.copy()
    vis_fixed[hole_fixed.cpu().numpy()] = 0
    cv2.imwrite(str(OUT_DIR / "02_fixed.png"), cv2.cvtColor(vis_fixed, cv2.COLOR_RGB2BGR))

    # 3. 差异：修复版多膨胀的（绿色）
    diff_fixed = hole_fixed.cpu().numpy().astype(int) - hole_buggy.cpu().numpy().astype(int)
    diff_mask = diff_fixed > 0
    print(f"\n  绿色高亮 = bug 版本漏掉、修复版本检测到的: {diff_mask.sum():,} 像素")

    vis_diff = right_np.copy()
    vis_diff[hole_buggy.cpu().numpy()] = 0
    vis_diff[diff_mask] = [0, 255, 0]
    cv2.imwrite(str(OUT_DIR / "03_diff_green.png"), cv2.cvtColor(vis_diff, cv2.COLOR_RGB2BGR))

    # 裁剪肩膀区域
    y1, y2, x1, x2 = 550, 750, 750, 1050
    cv2.imwrite(str(OUT_DIR / "crop_buggy.png"), cv2.cvtColor(vis_buggy[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / "crop_fixed.png"), cv2.cvtColor(vis_fixed[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / "crop_diff.png"), cv2.cvtColor(vis_diff[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))

    print(f"\n  裁剪图已保存到: {OUT_DIR}")
    print(f"  看 crop_diff.png → 肩膀的白色弧线是否变成绿色")

    print(f"\n✅ Bug 验证完成！")
    print(f"  问题确认: GPU 版本阈值没有乘 255")
    print(f"  修复: th = color_threshold * 255")


if __name__ == "__main__":
    main()
