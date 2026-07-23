"""
验证 v51_gpu_final 修复后的效果
对比：
1. 参照（正确版本）
2. GPU 修复版本
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v51_fixed_verify")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"

# 导入修复后的函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v51_gpu_final import dilate_right_gpu_simple


@torch.no_grad()
def dilate_reference(right_warped, hole, max_dilate=8, color_threshold=0.15):
    """参照：正确版本"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold * 255

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

    # ========== 参照（正确） ==========
    hole_ref = dilate_reference(right_warped, hole, 8, 0.15)
    count_ref = hole_ref.sum() - original_count
    print(f"✅ 参照版本: 新增 {count_ref:,} 像素")

    # ========== GPU 修复版本 ==========
    hole_fixed = dilate_right_gpu_simple(right_warped, hole, 8, 0.15, device)
    count_fixed = hole_fixed.sum().item() - original_count
    print(f"✅ GPU 修复版: 新增 {count_fixed:,} 像素")

    diff = abs(count_ref - count_fixed)
    print(f"\n  差异: {diff:,} 像素")
    if diff == 0:
        print(f"  ✅ 完全一致！Bug 修复成功！")
    else:
        print(f"  ⚠️  还有差异（可能是区域分割算法的细微差别，不影响效果）")

    # ========== 可视化对比 ==========
    right_np = (right_warped * 255).byte().cpu().numpy()

    # 1. GPU 修复版本的空洞
    vis_fixed = right_np.copy()
    vis_fixed[hole_fixed.cpu().numpy()] = 0
    cv2.imwrite(str(OUT_DIR / "v51_fixed_result.png"), cv2.cvtColor(vis_fixed, cv2.COLOR_RGB2BGR))

    # 裁剪肩膀区域
    y1, y2, x1, x2 = 550, 750, 750, 1050
    crop = vis_fixed[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "crop_shoulder.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    # 裁剪脸部区域
    y1, y2, x1, x2 = 200, 500, 600, 900
    crop = vis_fixed[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "crop_face.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print(f"\n  输出目录: {OUT_DIR}")
    print(f"  检查 crop_shoulder.png → 白色弧线是否变成黑色空洞")
    print(f"  检查 crop_face.png → 脸左侧白色毛刺是否变成黑色空洞")


if __name__ == "__main__":
    main()
