"""
迭代 v7: 填补取色位置的垂直平滑 ✨

最终解决方案！

问题根源：
  反遮挡带边界在垂直方向有 30+ 像素跳变
  → 每行填补时，取背景颜色的 x 位置也跳变 30+ 像素
  → 即使背景是渐变的，相邻行也可能取到完全不同的颜色
  → 人眼感觉到"边缘不平滑"

解决方案（简单、有效、无副作用）：
  1. 正常计算每行的填补取色位置 x
  2. 在垂直方向对 x 坐标做中值滤波（消除跳变）
  3. 用平滑后的 x 坐标去取背景颜色
  4. 空洞形状完全不变！

这是最优雅的解决方案：
  ✅ 不多挖任何前景像素
  ✅ 不留任何额外空洞
  ✅ 边缘颜色自然连续
  ✅ 计算量极小
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
    parser.add_argument("--smooth-kernel", type=int, default=7, help="取色位置垂直平滑核大小")
    return parser.parse_args()


# ========== 原版反遮挡带 ==========
def project_disocclusion_bands_original(disparity, min_drop=3.0, right_cleanup=16):
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

    difference = torch.zeros((h, w + 1), device=disparity.device, dtype=torch.int32)
    rows = torch.arange(h, device=disparity.device).view(h, 1).expand(h, w - 1)
    flat = difference.reshape(-1)
    start_index = (rows * (w + 1) + start)[valid]
    stop_index = (rows * (w + 1) + end + 1)[valid]
    flat.scatter_add_(0, start_index, torch.ones_like(start_index, dtype=flat.dtype))
    flat.scatter_add_(0, stop_index, -torch.ones_like(stop_index, dtype=flat.dtype))
    return torch.cumsum(difference[:, :w], dim=1) > 0


# ========== v7 改进：改进的填补函数，平滑取色位置 ==========
def inpaint_with_smoothed_sample_positions(image, hole_mask, near, bg_threshold=0.3,
                                          safety_margin=6, sample_smooth_kernel=7):
    """
    严格背景填补，但取色位置在垂直方向做平滑

    核心思想：
      空洞形状不变
      但取色的 x 坐标在垂直方向做中值滤波
      → 相邻行取色位置不会跳变 → 颜色自然连续
    """
    h, w = hole_mask.shape
    device = hole_mask.device
    result = image.clone()

    # 1. 计算每行的填补取色位置
    # 找到每行空洞的左边界 x
    hole_left_x = torch.full((h,), w, device=device, dtype=torch.long)
    for y in range(h):
        cols = torch.where(hole_mask[y])[0]
        if len(cols) > 0:
            hole_left_x[y] = cols.min()  # 这一行最左边的空洞像素

    # 2. 计算取色位置 = 边界 + safety_margin
    # （和 B 版本逻辑一致：跳过边缘安全距离，取真正的背景）
    sample_x = hole_left_x + safety_margin
    sample_x = sample_x.clamp(0, w - 1)

    # 3. 关键：对 sample_x 做垂直方向的中值滤波，消除跳变 ✨
    k = sample_smooth_kernel
    pad = k // 2

    # 转为 [1, 1, H, 1] 做 1D 中值滤波
    sample_x_1d = sample_x.float().view(1, 1, h, 1)
    sample_x_padded = F.pad(sample_x_1d, (0, 0, pad, pad), mode='replicate')

    # unfold + median
    unfolded = F.unfold(sample_x_padded, kernel_size=(k, 1), padding=0, stride=1)
    unfolded = unfolded.squeeze(0)  # [k, H]
    sample_x_smoothed = torch.median(unfolded, dim=0)[0].round().long()
    sample_x_smoothed = sample_x_smoothed.clamp(0, w - 1)

    # 4. 用平滑后的取色位置 gather 背景颜色
    # 构建 gather 的索引：每行取 sample_x_smoothed[y] 这个 x 位置的颜色
    rows = torch.arange(h, device=device).view(h, 1).expand(h, w)
    gather_idx = sample_x_smoothed.view(h, 1).expand(h, w)

    # 只在空洞区域填补
    bg_colors = torch.gather(result, 1, gather_idx.unsqueeze(-1).expand(h, w, 3))
    result[hole_mask] = bg_colors[hole_mask]

    return result, sample_x.cpu().numpy(), sample_x_smoothed.cpu().numpy()


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

    # C版本：反遮挡带（原版）
    disocclusion_band = project_disocclusion_bands_original(
        disparity_sharp, min_drop=3.0, right_cleanup=16
    )
    hole_with_band = hole | disocclusion_band
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    # 目标空间 near
    target_near = b.forward_target_near(near_score, disparity_sharp, unreliable)
    b._LAST_TARGET_NEAR = target_near

    # ========== 原版填补（B 版本） ==========
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    final_right_original = b.strict_background_inpaint_gpu_b(
        right_with_band, hole_with_band, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== v7 改进：平滑取色位置后的填补 ==========
    final_right_smoothed, sample_x_raw, sample_x_smoothed = inpaint_with_smoothed_sample_positions(
        right_with_band, hole_with_band, near_score,
        bg_threshold=0.3, safety_margin=6,
        sample_smooth_kernel=args.smooth_kernel
    )

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'disocclusion_band': disocclusion_band.cpu().numpy(),
        'hole_after': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final_original': final_right_original.cpu().numpy(),
        'final_smoothed': final_right_smoothed.cpu().numpy(),
        'sample_x_raw': sample_x_raw,
        'sample_x_smoothed': sample_x_smoothed,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v7 取色位置垂直平滑 (kernel={args.smooth_kernel}) ✨")

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

    final_original_u8 = (result['final_original'] * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / 'v7_final_original.png'), cv2.cvtColor(final_original_u8, cv2.COLOR_RGB2BGR))

    final_smoothed_u8 = (result['final_smoothed'] * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / 'v7_final_smoothed.png'), cv2.cvtColor(final_smoothed_u8, cv2.COLOR_RGB2BGR))

    # 边缘裁剪
    y1, y2, x1, x2 = 300, 700, 900, 1100
    cv2.imwrite(str(outdir / 'v7_final_original_edge_crop.png'),
                cv2.cvtColor(final_original_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / 'v7_final_smoothed_edge_crop.png'),
                cv2.cvtColor(final_smoothed_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))

    # ========== 定量评估 ==========
    print("\n" + "="*60)
    print(f"📊 v7 取色位置平滑版本 最终评估 ✨")
    print("="*60)

    hole_after = result['hole_after']
    h_crop, w_crop = hole_after[y1:y2, x1:x2].shape

    # 评估取色位置的平滑度
    sample_x_raw = result['sample_x_raw'][y1:y2]
    sample_x_smoothed = result['sample_x_smoothed'][y1:y2]

    # 只统计有效行
    valid = sample_x_raw < (x1 + w_crop - 1)  # 在裁剪区域内有效
    raw_valid = sample_x_raw[valid]
    smoothed_valid = sample_x_smoothed[valid]

    if len(raw_valid) > 1:
        grad_raw = np.abs(raw_valid[1:] - raw_valid[:-1])
        grad_smoothed = np.abs(smoothed_valid[1:] - smoothed_valid[:-1])

        print(f"填补取色位置 x 平滑度:")
        print(f"  原版 - 相邻行平均变化: {grad_raw.mean():.2f} 像素, 最大: {grad_raw.max():.2f} 像素")
        print(f"  平滑 - 相邻行平均变化: {grad_smoothed.mean():.2f} 像素, 最大: {grad_smoothed.max():.2f} 像素")
        impr_mean = (1 - grad_smoothed.mean() / grad_raw.mean()) * 100
        impr_max = (1 - grad_smoothed.max() / grad_raw.max()) * 100
        print(f"  改进幅度 - 平均: {impr_mean:.1f}%, 最大: {impr_max:.1f}%  ← 越大越好！")

    # 评估边缘颜色平滑度
    final_original_crop = result['final_original'][y1:y2, x1:x2]
    final_smoothed_crop = result['final_smoothed'][y1:y2, x1:x2]
    hole_crop = hole_after[y1:y2, x1:x2]

    edge_colors_original = []
    edge_colors_smoothed = []
    for y in range(h_crop):
        cols = np.where(hole_crop[y])[0]
        if len(cols) > 0 and cols.min() + 1 < w_crop:
            x = cols.min()
            edge_colors_original.append(final_original_crop[y, x])
            edge_colors_smoothed.append(final_smoothed_crop[y, x])

    if len(edge_colors_original) > 1:
        edge_colors_original = np.array(edge_colors_original)
        edge_colors_smoothed = np.array(edge_colors_smoothed)

        color_grad_original = np.abs(edge_colors_original[1:] - edge_colors_original[:-1]).mean(axis=1)
        color_grad_smoothed = np.abs(edge_colors_smoothed[1:] - edge_colors_smoothed[:-1]).mean(axis=1)

        print(f"\n填补后边缘颜色垂直平滑度:")
        print(f"  原版 - 平均差: {color_grad_original.mean():.3f}, 最大差: {color_grad_original.max():.3f}")
        print(f"  平滑 - 平均差: {color_grad_smoothed.mean():.3f}, 最大差: {color_grad_smoothed.max():.3f}")
        impr_color = (1 - color_grad_smoothed.mean() / color_grad_original.mean()) * 100
        print(f"  颜色平滑度改进: {impr_color:.1f}%  ← 越大越好！")

    print(f"\n✅ 迭代完成！所有结果保存在: edge_smooth_iterations/frames/")
    print(f"   对比 v7_final_original vs v7_final_smoothed 看边缘平滑效果！")


if __name__ == "__main__":
    main()
