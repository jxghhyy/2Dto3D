"""
迭代 v5: 视差边缘的垂直平滑

关键洞察！
  之前 v1-v4 都在下游修修补补，但真正的根源是：
    视差边缘本身在垂直方向就是不连续的（逐行独立）
    → 反遮挡带边界也是锯齿状的
    → 填补时每行取的背景颜色来源不一样
    → 最终边缘颜色跳变

正确方法：
  在反遮挡带计算之前，对视差边缘做垂直平滑！
  让视差边缘本身就是平滑曲线 → 反遮挡带边界平滑 → 填补颜色来源连续 → 最终边缘平滑
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
    parser.add_argument("--smooth-kernel", type=int, default=7, help="视差边缘垂直平滑核大小")
    return parser.parse_args()


# ========== v5 改进：视差边缘的垂直平滑 ==========
def project_disocclusion_bands_smooth_disparity(disparity, min_drop=3.0, right_cleanup=16, smooth_kernel=7):
    """
    关键改进：先平滑视差边缘，再计算反遮挡带

    步骤：
    1. 找到每行的视差下降边缘位置（前景边界）
    2. 对这些边界位置做垂直方向的中值滤波（得到平滑曲线）
    3. 用平滑后的边界位置计算反遮挡带
    """
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    device = disparity.device

    # ========== 步骤 1：找到每行的视差下降边缘 ==========
    # 视差梯度（向右）
    disp_grad = disparity[:, :-1] - disparity[:, 1:]
    is_drop = disp_grad >= min_drop  # 视差显著下降 = 前景边缘

    # 找到每行最左边的下降位置（主边缘）
    edge_x = torch.full((h,), w, device=device, dtype=torch.long)
    for y in range(h):
        drop_positions = torch.where(is_drop[y])[0]
        if len(drop_positions) > 0:
            edge_x[y] = drop_positions.min()  # 最左边的下降（主边缘）

    # ========== 步骤 2：垂直方向平滑边缘位置 ==========
    # 先剔除离群点
    valid_mask = edge_x < w - 1
    if valid_mask.sum() > 10:
        valid_values = edge_x[valid_mask].float()
        median = torch.median(valid_values)
        mad = torch.median(torch.abs(valid_values - median))
        threshold = max(3.0 * mad, 5.0)  # 至少 5 像素阈值

        outlier_mask = torch.abs(edge_x.float() - median) > threshold
        edge_x[outlier_mask] = median.round().long()

    # 中值滤波（垂直方向）
    pad = smooth_kernel // 2
    edge_1d = edge_x.float().view(1, 1, h, 1)
    edge_padded = F.pad(edge_1d, (0, 0, pad, pad), mode='replicate')

    # unfold + median
    unfolded = F.unfold(edge_padded, kernel_size=(smooth_kernel, 1), padding=0, stride=1)
    unfolded = unfolded.squeeze(0)
    edge_smoothed = torch.median(unfolded, dim=0)[0].round().long()
    edge_smoothed = edge_smoothed.clamp(0, w - 1)

    # ========== 步骤 3：用平滑后的边界计算反遮挡带 ==========
    # 每个像素到边缘的距离：edge_smoothed[y] 是这一行的边缘 x 坐标
    # 反遮挡带从 edge_smoothed[y] + 1 开始，向右延伸 right_cleanup 像素
    x_range = torch.arange(w, device=device).view(1, w).expand(h, w)
    dist_from_edge = x_range - edge_smoothed.view(h, 1)

    # 这一行边缘的视差值
    edge_disp = disparity[torch.arange(h, device=device), edge_smoothed.clamp(0, w-1)]

    # 用视差值计算理论上应该的反遮挡宽度（right_cleanup 是默认值）
    # 简单起见：边缘右边 right_cleanup 像素都是反遮挡带
    band = (dist_from_edge >= 0) & (dist_from_edge <= right_cleanup)

    return band


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

    # C版本：v5 改进 - 视差边缘平滑后的反遮挡带
    disocclusion_band = project_disocclusion_bands_smooth_disparity(
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
    print(f"设备: {device}, 版本: v5 视差边缘垂直平滑 (kernel={args.smooth_kernel})")

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
    cv2.imwrite(str(outdir / 'v5_final_right.png'), cv2.cvtColor(final_u8, cv2.COLOR_RGB2BGR))

    band_viz = (result['warped'] * 255).astype(np.uint8)
    band_viz[result['disocclusion_band']] = [0, 255, 0]
    cv2.imwrite(str(outdir / 'v5_band_green.png'), cv2.cvtColor(band_viz, cv2.COLOR_RGB2BGR))

    # 边缘裁剪
    y1, y2, x1, x2 = 300, 700, 900, 1100
    cv2.imwrite(str(outdir / 'v5_final_edge_crop.png'),
                cv2.cvtColor(final_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / 'v5_band_edge_crop.png'),
                cv2.cvtColor(band_viz[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))

    # ========== 定量评估 ==========
    print("\n" + "="*60)
    print(f"📊 v5 视差边缘平滑版本 评估")
    print("="*60)

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

    # 分析填补后边缘颜色的垂直连续性
    final_crop = result['final'][y1:y2, x1:x2]
    edge_colors = []
    for y in range(h_crop):
        cols = np.where(hole_crop[y])[0]
        if len(cols) > 0 and cols.min() + 1 < w_crop:
            x = cols.min()
            edge_colors.append(final_crop[y, x])

    if len(edge_colors) > 1:
        edge_colors = np.array(edge_colors)
        color_grad = np.abs(edge_colors[1:] - edge_colors[:-1]).mean(axis=1)
        print(f"\n填补后边缘颜色垂直连续性:")
        print(f"  相邻行平均颜色差: {color_grad.mean():.3f}")
        print(f"  相邻行最大颜色差: {color_grad.max():.3f}")

    print("\nv5 完成！结果保存到:", outdir)


if __name__ == "__main__":
    main()
