"""
测试 v46 自适应膨胀在单帧上的效果
对比：固定膨胀4 vs 自适应膨胀max=5
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


@torch.no_grad()
def dilate_hole_right_fixed(hole, dilate_pixels, device):
    """旧版：固定膨胀量"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()

    for y in range(h):
        row = hole_np[y]
        if not row.any():
            continue
        indices = np.where(row)[0]
        regions = []
        if len(indices) > 0:
            start_x = indices[0]
            prev_x = indices[0]
            for i in range(1, len(indices)):
                x = indices[i]
                if x > prev_x + 1:
                    regions.append((start_x, prev_x))
                    start_x = x
                prev_x = x
            regions.append((start_x, prev_x))
        for (start_x, end_x) in regions:
            new_end_x = min(w - 1, end_x + dilate_pixels)
            hole_dilated[y, end_x:new_end_x + 1] = True

    return torch.from_numpy(hole_dilated).to(device)


@torch.no_grad()
def dilate_hole_right_adaptive(hole, max_dilate, max_disparity, device):
    """新版：自适应膨胀量"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()

    for y in range(h):
        row = hole_np[y]
        if not row.any():
            continue
        indices = np.where(row)[0]
        regions = []
        if len(indices) > 0:
            start_x = indices[0]
            prev_x = indices[0]
            for i in range(1, len(indices)):
                x = indices[i]
                if x > prev_x + 1:
                    regions.append((start_x, prev_x))
                    start_x = x
                prev_x = x
            regions.append((start_x, prev_x))
        for (start_x, end_x) in regions:
            width = end_x - start_x + 1
            dilate_pixels = int(round(width / max_disparity * max_dilate))
            dilate_pixels = max(1, min(max_dilate, dilate_pixels))
            new_end_x = min(w - 1, end_x + dilate_pixels)
            hole_dilated[y, end_x:new_end_x + 1] = True

    return torch.from_numpy(hole_dilated).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

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

    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    max_disparity = 24.0 * w_orig / w_orig

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
    disparity = near_score * max_disparity

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    original_count = hole.sum().item()
    print(f"原始空洞: {original_count} 像素")

    # ========== 对比三种方案 ==========
    out_dir = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/adaptive_dilate_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 不膨胀
    vis_none = (right_warped * 255).byte().cpu().numpy().copy()
    vis_none[hole.cpu().numpy()] = 0
    cv2.imwrite(str(out_dir / "01_no_dilate.png"), cv2.cvtColor(vis_none, cv2.COLOR_RGB2BGR))

    # 2. 固定膨胀4
    hole_fixed4 = dilate_hole_right_fixed(hole, 4, device)
    fixed4_count = hole_fixed4.sum().item()
    vis_fixed4 = (right_warped * 255).byte().cpu().numpy().copy()
    vis_fixed4[hole_fixed4.cpu().numpy()] = 0
    cv2.imwrite(str(out_dir / "02_fixed_dilate4.png"), cv2.cvtColor(vis_fixed4, cv2.COLOR_RGB2BGR))

    # 3. 自适应膨胀max=5
    hole_adaptive = dilate_hole_right_adaptive(hole, 5, 24.0, device)
    adaptive_count = hole_adaptive.sum().item()
    vis_adaptive = (right_warped * 255).byte().cpu().numpy().copy()
    vis_adaptive[hole_adaptive.cpu().numpy()] = 0
    cv2.imwrite(str(out_dir / "03_adaptive_max5.png"), cv2.cvtColor(vis_adaptive, cv2.COLOR_RGB2BGR))

    print(f"\n膨胀效果对比:")
    print(f"  不膨胀:    {original_count} 像素")
    print(f"  固定膨胀4:  {fixed4_count} 像素 (+{fixed4_count - original_count})")
    print(f"  自适应max5: {adaptive_count} 像素 (+{adaptive_count - original_count})")

    # ========== 差异可视化：自适应比固定少膨胀了哪些地方 ==========
    diff_np = hole_fixed4.cpu().numpy() & ~hole_adaptive.cpu().numpy()
    print(f"  自适应比固定少膨胀了 {diff_np.sum()} 像素")

    diff_vis = (right_warped * 255).byte().cpu().numpy().copy()
    diff_vis[diff_np] = [255, 0, 0]  # 红色 = 固定膨胀4有，但自适应没有的（省掉的区域）
    cv2.imwrite(str(out_dir / "04_diff_fixed4_vs_adaptive_red.png"),
               cv2.cvtColor(diff_vis, cv2.COLOR_RGB2BGR))

    # 裁剪放大球拍附近
    x1, x2 = 800, 900
    y1, y2 = 200, 400
    crop_fixed = vis_fixed4[y1:y2, x1:x2]
    crop_adaptive = vis_adaptive[y1:y2, x1:x2]
    crop_diff = diff_vis[y1:y2, x1:x2]

    crop_fixed_bgr = cv2.cvtColor(crop_fixed, cv2.COLOR_RGB2BGR)
    crop_adaptive_bgr = cv2.cvtColor(crop_adaptive, cv2.COLOR_RGB2BGR)
    crop_diff_bgr = cv2.cvtColor(crop_diff, cv2.COLOR_RGB2BGR)

    crop_fixed_x4 = cv2.resize(crop_fixed_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    crop_adaptive_x4 = cv2.resize(crop_adaptive_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    crop_diff_x4 = cv2.resize(crop_diff_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(out_dir / "05_crop_fixed4_x4.png"), crop_fixed_x4)
    cv2.imwrite(str(out_dir / "06_crop_adaptive_x4.png"), crop_adaptive_x4)
    cv2.imwrite(str(out_dir / "07_crop_diff_red_x4.png"), crop_diff_x4)

    print(f"\n✅ 对比图已保存到: {out_dir}")
    print(f"  01_no_dilate.png - 不膨胀（原始空洞）")
    print(f"  02_fixed_dilate4.png - 固定膨胀4")
    print(f"  03_adaptive_max5.png - 自适应膨胀max=5")
    print(f"  04_diff_..._red.png - 红色=自适应少膨胀的区域（小洞）")
    print(f"  05-07 球拍附近放大4倍")


if __name__ == "__main__":
    main()
