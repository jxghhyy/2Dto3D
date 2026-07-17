"""
迭代 v1: 垂直平滑反遮挡带
在原版基础上，对反遮挡带的左右边界做垂直方向的平滑
"""
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, '.')
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--smooth-kernel", type=int, default=5, help="垂直平滑核大小")
    return parser.parse_args()


# ========== v1 改进：垂直平滑反遮挡带 ==========
def project_disocclusion_bands_smoothed(disparity, min_drop=3.0, right_cleanup=16, smooth_kernel=5):
    """
    v1 改进版：
    1. 先生成原始的反遮挡带（和原版一样）
    2. 提取反遮挡带的左边界坐标
    3. 对左边界做垂直方向的平滑滤波
    4. 用平滑后的边界重新生成反遮挡带
    """
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    x_left = torch.arange(w - 1, device=disparity.device).view(1, w - 1)
    d_left = disparity[:, :-1]
    d_right = disparity[:, 1:]
    is_drop = (d_left - d_right) >= min_drop

    foreground_target = x_left.to(disparity.dtype) - d_left
    background_target = (x_left + 1).to(disparity.dtype) - d_right
    start = torch.floor(foreground_target).long() + 1
    end = torch.floor(background_target).long() + right_cleanup
    start = start.clamp(0, w - 1)
    end = end.clamp(0, w - 1)
    valid = is_drop & (end >= start)

    # ========== 关键点：提取边界后平滑 ==========
    # 生成原始的 difference array
    difference = torch.zeros((h, w + 1), device=disparity.device, dtype=torch.float32)
    rows = torch.arange(h, device=disparity.device).view(h, 1).expand(h, w - 1)
    start_index = (rows * (w + 1) + start)[valid]
    stop_index = (rows * (w + 1) + end + 1)[valid]
    difference.reshape(-1).scatter_add_(0, start_index, torch.ones_like(start_index, dtype=torch.float32))
    difference.reshape(-1).scatter_add_(0, stop_index, -torch.ones_like(stop_index, dtype=torch.float32))

    # 得到原始的带掩码
    band_raw = torch.cumsum(difference[:, :w], dim=1) > 0

    # ========== 提取左边界 ==========
    # 找到每行第一个带像素的位置（左边界）
    band_float = band_raw.float()
    band_shift_right = F.pad(band_float[:, :-1], (1, 0))  # 右边补0，左边移一位
    left_edge = (band_float - band_shift_right) > 0.5  # 从 0 变 1 的位置就是左边界

    # 提取左边界的 x 坐标
    y_coords, x_coords = torch.where(left_edge)
    edge_map = torch.full((h,), w, device=disparity.device, dtype=torch.long)
    edge_map[y_coords] = x_coords  # 每行的左边界 x 坐标

    # ========== 垂直平滑边界 ==========
    # 用 1D 卷积做垂直方向平滑
    edge_float = edge_map.float().view(1, 1, h, 1)  # [N,C,H,W] 格式
    kernel_size = smooth_kernel
    padding = kernel_size // 2

    # 高斯核（比平均核更好，边缘不会糊）
    sigma = kernel_size / 3.0
    kernel_1d = torch.exp(-(torch.arange(kernel_size, device=disparity.device, dtype=torch.float32) - padding) ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d.view(1, 1, kernel_size, 1)  # 垂直方向 1D 核

    edge_smoothed = F.conv2d(edge_float, kernel, padding=(padding, 0))[0, 0, :, 0].round().long()
    edge_smoothed = edge_smoothed.clamp(0, w - 1)

    # ========== 用平滑后的边界重新生成带 ==========
    x_range = torch.arange(w, device=disparity.device).view(1, w).expand(h, w)
    band_smoothed = x_range >= edge_smoothed.view(h, 1)

    # 右边界还是用原来的（right_cleanup 保持不变）
    # 但我们也可以对右边界做同样的平滑
    # （暂时只平滑左边界，因为右边界是宽度，不影响边缘形状）

    return band_smoothed


# ========== 完整管道 ==========
def process_frame(frame_bgr, model, device, args):
    h_orig, w_orig = frame_bgr.shape[:2]

    # 深度推理
    img = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)
    img_resized = F.interpolate(img, size=(294, 518), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    model_input = (img_resized - mean_t) / std_t

    with torch.no_grad():
        depth_raw = model(model_input)[0].float()

    # 归一化
    flat = depth_raw.reshape(-1)
    idx = torch.randint(0, flat.numel(), (16384,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=flat.device))
    low, high = q_vals[0], q_vals[1]
    depth_norm = ((depth_raw - low) / (high - low)).clamp(0.0, 1.0)

    # 上采样
    near_score = F.interpolate(
        depth_norm[None, None, :, :],
        size=(h_orig, w_orig),
        mode="bilinear", align_corners=False
    )[0, 0]

    disparity = near_score * 24.0

    # B版本：锐化 + 排除过渡像素
    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    # C版本：v1 改进的平滑反遮挡带
    disocclusion_band = project_disocclusion_bands_smoothed(
        disparity_sharp, min_drop=3.0, right_cleanup=16, smooth_kernel=args.smooth_kernel
    )
    hole_with_band = hole | disocclusion_band
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    # 目标空间 near
    target_near = b.forward_target_near(near_score, disparity_sharp, unreliable)
    b._LAST_TARGET_NEAR = target_near

    # 填补
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    final_right = b.strict_background_inpaint_gpu_b(
        right_with_band, hole_with_band, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'disocclusion_band': disocclusion_band.cpu().numpy(),
        'hole_before': hole.cpu().numpy(),
        'hole_after': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final': final_right.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v1 垂直平滑 (kernel={args.smooth_kernel})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # 只处理第 240 帧
    cap = cv2.VideoCapture(args.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 240)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        print("无法读取第240帧")
        return

    print("处理第240帧...")
    result = process_frame(frame_bgr, model, device, args)

    # 保存结果
    h, w = frame_bgr.shape[:2]

    final_u8 = (result['final'] * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / 'v1_final_right.png'), cv2.cvtColor(final_u8, cv2.COLOR_RGB2BGR))

    band_viz = (result['warped'] * 255).astype(np.uint8)
    band_viz[result['disocclusion_band']] = [0, 255, 0]
    cv2.imwrite(str(outdir / 'v1_band_green.png'), cv2.cvtColor(band_viz, cv2.COLOR_RGB2BGR))

    hole_viz = (final_u8.copy() * 0.5).astype(np.uint8)
    hole_viz[result['hole_after']] = [0, 0, 255]
    cv2.imwrite(str(outdir / 'v1_hole_red.png'), cv2.cvtColor(hole_viz, cv2.COLOR_RGB2BGR))

    # 边缘裁剪
    y1, y2, x1, x2 = 300, 700, 900, 1100
    cv2.imwrite(str(outdir / 'v1_final_edge_crop.png'),
                cv2.cvtColor(final_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / 'v1_band_edge_crop.png'),
                cv2.cvtColor(band_viz[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))

    # ========== 定量评估 ==========
    print("\n" + "="*60)
    print(f"📊 v1 垂直平滑版本 (kernel={args.smooth_kernel}) 边缘平滑度评估")
    print("="*60)

    band = result['disocclusion_band']
    hole_after = result['hole_after']

    h_crop, w_crop = hole_after[y1:y2, x1:x2].shape
    hole_crop = hole_after[y1:y2, x1:x2]

    edge_x = np.zeros(h_crop, dtype=int)
    for y in range(h_crop):
        cols = np.where(hole_crop[y])[0]
        if len(cols) > 0:
            edge_x[y] = cols.min()

    valid = edge_x > 0
    edge_x_valid = edge_x[valid]

    if len(edge_x_valid) > 0:
        grad = np.abs(edge_x_valid[1:] - edge_x_valid[:-1])
        print(f"空洞左边界 x 坐标统计:")
        print(f"  平均值: {edge_x_valid.mean():.1f}")
        print(f"  标准差: {edge_x_valid.std():.2f}  ← 越小越平滑")
        print(f"  相邻行平均变化: {grad.mean():.2f} 像素")
        print(f"  相邻行最大变化: {grad.max():.2f} 像素  ← 越小越平滑")

        if len(grad) > 1:
            second_grad = np.abs(grad[1:] - grad[:-1])
            print(f"  二阶差分平均: {second_grad.mean():.2f}")

    print("\nv1 完成！结果保存到:", outdir)


if __name__ == "__main__":
    main()
