"""
分析第5次迭代后剩下的空洞的左边界
判断左侧是否都是背景（可以安全从左往右填）
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


def create_strict_right_kernel(kernel_size, device):
    pad = kernel_size // 2
    x_indices = torch.arange(kernel_size, device=device) - pad
    horizontal_mask = x_indices >= 0
    horizontal_weights = torch.ones(kernel_size, device=device, dtype=torch.float32)
    horizontal_weights[~horizontal_mask] = 0.0
    kernel_1d = horizontal_weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    num_nonzero = horizontal_weights.sum().item() * kernel_size
    if num_nonzero > 0:
        kernel_2d = kernel_2d / kernel_2d.sum() * num_nonzero
    return kernel_2d


@torch.no_grad()
def dilate_hole_right_fixed(hole, dilate_pixels, device):
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
def inpaint_strict_right_n_iters(img, hole, kernel, k3, n_iters):
    """只跑 n 次迭代，返回中间状态"""
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2
    device = hole.device

    for it in range(n_iters):
        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, kernel, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
        filled_count = can_fill.sum().item()

        if filled_count == 0:
            break

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

    return result, hole_cur


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

    # 深度推理参数
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

    # 深度推理
    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

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

    # 向右膨胀 4 像素
    hole_dilated = dilate_hole_right_fixed(hole, 4, device)
    right_with_hole = right_warped.clone()
    right_with_hole[hole_dilated] = 0.0

    print(f"初始空洞: {hole_dilated.sum().item():,} 像素")

    kernel = create_strict_right_kernel(15, device)
    k3 = kernel.repeat(3, 1, 1, 1)

    # 跑 5 次迭代
    result_after_5, hole_after_5 = inpaint_strict_right_n_iters(
        right_with_hole, hole_dilated, kernel, k3, n_iters=5
    )

    remaining_5 = hole_after_5.sum().item()
    print(f"第 5 次迭代后剩余: {remaining_5:,} 像素")

    # ========== 分析剩余空洞的左边界 ==========
    hole_np = hole_after_5.cpu().numpy()
    result_np = result_after_5.cpu().numpy()

    left_pixels = []  # 空洞左边界像素的颜色 (RGB)
    left_x_coords = []

    for y in range(h_orig):
        row = hole_np[y]
        if not row.any():
            continue
        indices = np.where(row)[0]
        left_x = indices[0]  # 空洞的左边界
        if left_x > 0:
            left_pixels.append(result_np[y, left_x - 1])
            left_x_coords.append(left_x)

    left_pixels = np.array(left_pixels)
    print(f"\n分析 {len(left_pixels)} 个空洞行的左边界:")

    # 颜色统计（绿色通道高的是草地）
    print(f"  左边界像素颜色统计:")
    print(f"    R 均值: {left_pixels[:,0].mean():.3f} ± {left_pixels[:,0].std():.3f}")
    print(f"    G 均值: {left_pixels[:,1].mean():.3f} ± {left_pixels[:,1].std():.3f}")
    print(f"    B 均值: {left_pixels[:,2].mean():.3f} ± {left_pixels[:,2].std():.3f}")

    # 绿色显著高于红蓝色 = 草地
    is_grass = (left_pixels[:,1] > left_pixels[:,0]) & (left_pixels[:,1] > left_pixels[:,2])
    grass_pct = is_grass.mean() * 100
    print(f"\n  左边界是草地（G > R 且 G > B）的比例: {grass_pct:.1f}%")

    # 左边界 x 坐标分布
    print(f"  左边界 x 均值: {np.mean(left_x_coords):.1f}")
    print(f"  左边界 x 中位: {np.median(left_x_coords):.1f}")
    print(f"  左边界 x 范围: {min(left_x_coords)} ~ {max(left_x_coords)}")

    # ========== 可视化验证 ==========
    out_dir = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/hole_left_boundary_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 第5次迭代后的图，空洞左边界标红
    vis_5 = (result_after_5 * 255).byte().cpu().numpy().copy()
    for y in range(h_orig):
        row = hole_np[y]
        if not row.any():
            continue
        left_x = np.where(row)[0][0]
        # 左边界 3 像素标红
        vis_5[y, max(0, left_x-2):left_x+1] = [255, 0, 0]

    cv2.imwrite(str(out_dir / "01_iter5_hole_left_boundary_red.png"),
               cv2.cvtColor(vis_5, cv2.COLOR_RGB2BGR))

    # 裁剪放大
    x1, x2 = 1850, 1920
    y1, y2 = 300, 800
    crop = vis_5[y1:y2, x1:x2]
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    crop_bgr_x4 = cv2.resize(crop_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "02_crop_x4.png"), crop_bgr_x4)

    print(f"\n✅ 分析图已保存到: {out_dir}")
    print(f"  红色标记 = 空洞的左边界（第5次迭代后）")
    print(f"  看颜色是不是都是绿色草地！")


if __name__ == "__main__":
    main()
