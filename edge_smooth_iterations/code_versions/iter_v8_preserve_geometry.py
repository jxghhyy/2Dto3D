"""
迭代 v8: 保留背景几何结构的平滑填补 ✨

问题分析：
1. v7 的问题：每行用同一个 x 填补整行，破坏了 B 版本的镜像机制
   - 结果：天花板灯等几何结构被扭曲了

2. 反遮挡带过度扩散：right_cleanup=16 可能太大
   - 结果：手部等前景细节被错误地标记为空洞

解决方案：
1. 保留 B 版本的镜像机制（每个像素取不同的背景位置）
2. 但对取色的 x 坐标场做垂直方向的平滑（而不是标量）
3. 减小反遮挡带的 right_cleanup 参数
4. 添加深度梯度约束，避免跨越深度边缘取色
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
    parser.add_argument("--smooth-kernel", type=int, default=5, help="取色位置垂直平滑核大小")
    parser.add_argument("--band-cleanup", type=int, default=8, help="反遮挡带清理宽度（减小减少扩散）")
    parser.add_argument("--min-drop", type=float, default=4.0, help="视差下降阈值（增大减少误判）")
    return parser.parse_args()


# ========== v8 改进：更保守的反遮挡带 ==========
def project_disocclusion_bands_conservative(disparity, min_drop=4.0, right_cleanup=8):
    """
    更保守的反遮挡带计算：
    1. 提高视差下降阈值，减少误判
    2. 减小 right_cleanup，避免过度扩散
    3. 添加垂直方向的连续性约束
    """
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    x_left = torch.arange(w - 1, device=disparity.device).view(1, w - 1)
    d_left = disparity[:, :-1]
    d_right = disparity[:, 1:]

    # 1. 提高阈值，只有明显的视差下降才认为是遮挡边缘
    is_drop = (d_left - d_right) >= min_drop

    # 2. 添加梯度约束：只有局部梯度大的地方才认为是边缘
    disp_grad = torch.abs(d_left - d_right)
    grad_mean = disp_grad.mean()
    grad_std = disp_grad.std()
    is_significant_edge = disp_grad >= (grad_mean + 0.5 * grad_std)
    is_drop = is_drop & is_significant_edge

    foreground_target = x_left.to(disparity.dtype) - d_left
    background_target = (x_left + 1).to(disparity.dtype) - d_right
    start = torch.floor(foreground_target).long() + 1
    end = torch.floor(background_target).long() + right_cleanup  # 减小 cleanup
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

    band = torch.cumsum(difference[:, :w], dim=1) > 0

    # 3. 后处理：移除太小的孤立带
    if h > 5:
        band_v = band.float()
        # 垂直闭运算连接相邻的带
        k = 3
        pad = k // 2
        kernel = torch.ones((1, 1, k, 1), device=band.device)
        band_dilated = F.conv2d(band_v.view(1, 1, h, w), kernel, padding=(pad, 0)) > 0.5
        band_closed = F.conv2d(band_dilated.float(), kernel, padding=(pad, 0)) > 0.5
        band = band_closed[0, 0]

    return band


# ========== v8 改进：保留镜像机制的垂直平滑 ==========
def inpaint_with_smoothed_mirroring(image, hole_mask, target_near, bg_threshold=0.3,
                                    safety_margin=6, smooth_kernel=5):
    """
    核心改进：保留 B 版本的镜像机制，同时对取色坐标做垂直平滑

    B 版本逻辑：
    - 每个空洞像素 x 找到右边最近的背景像素 right_index
    - 然后镜像偏移：sample_x = right_index + mirror_offset
    - 结果：宽空洞中不同像素取不同背景位置，保留了几何结构

    v8 改进：
    - 计算出 sample_x 场后，在垂直方向做中值滤波
    - 既保留几何结构，又保证颜色连续
    """
    h, w = hole_mask.shape
    device = hole_mask.device
    result = image.clone()

    x = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
    known = (~hole_mask) & (target_near >= 0.0)

    # ========== 1. 安全边缘：和 B 版本一样 ==========
    right_bg = known.clone()
    for offset in range(1, safety_margin + 1):
        has_bg_left = torch.zeros_like(known)
        has_bg_left[:, offset:] = known[:, :-offset]
        right_bg &= has_bg_left

    # ========== 2. 找到右边最近的有效背景：和 B 版本一样 ==========
    right_seed = torch.where(right_bg, x, torch.full_like(x, w))
    right_index = torch.flip(
        torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    right_distance = right_index - x
    right_ok_geometry = (
        (right_index < w)
        & (right_distance >= 0)
        & (right_distance <= 200)  # strict_bg_max_distance
    )

    # ========== 3. 深度验证：和 B 版本一样 ==========
    raw_left_seed = torch.where(known, x, torch.full_like(x, -1))
    raw_left_index = torch.cummax(raw_left_seed, dim=1).values
    left_boundary_exists = raw_left_index >= 0

    left_boundary_near = torch.gather(
        target_near, 1, raw_left_index.clamp(0, w - 1)
    )
    right_boundary_near = torch.gather(
        target_near, 1, right_index.clamp(0, w - 1)
    )
    right_depth_ok = (
        (~left_boundary_exists)
        | (right_boundary_near <= left_boundary_near + 0.025)
    )

    # ========== 4. 镜像偏移：和 B 版本一样（关键！保留几何结构） ==========
    mirror_offset = (right_distance - safety_margin - 1).clamp_min(0)
    mirror_budget = (200 - right_distance).clamp_min(0)
    mirror_offset = torch.minimum(mirror_offset, mirror_budget)

    # 这是关键：每个像素有独立的取色位置！
    sample_x = (right_index + mirror_offset).clamp(0, w - 1)

    # ========== 5. ✨ v8 新增：垂直方向平滑 sample_x 场 ✨ ==========
    # 不是对每行的单个值做平滑，而是对整个 2D sample_x 场做垂直中值滤波
    k = smooth_kernel
    pad = k // 2

    # 转换为 [1, 1, H, W] 做 2D 滤波（只在垂直方向）
    sample_x_float = sample_x.float().view(1, 1, h, w)

    # unfold 做中值滤波
    # 填充边界
    sample_x_padded = F.pad(sample_x_float, (0, 0, pad, pad), mode='replicate')

    # 只在垂直方向做中值滤波（核大小 k x 1）
    # unfold: [1, kernel_size, num_patches]
    unfolded = F.unfold(sample_x_padded, kernel_size=(k, 1), padding=0, stride=1)
    unfolded = unfolded.view(k, h, w)  # [k, H, W]

    # 中值滤波消除垂直方向的跳变
    sample_x_smoothed = torch.median(unfolded, dim=0)[0].round().long()
    sample_x_smoothed = sample_x_smoothed.clamp(0, w - 1)

    # ========== 6. 验证取色位置的深度：和 B 版本一样 ==========
    sample_right_known = torch.gather(known, 1, sample_x_smoothed)
    sample_right_near = torch.gather(target_near, 1, sample_x_smoothed)
    sample_right_depth_ok = (
        (~left_boundary_exists)
        | (sample_right_near <= left_boundary_near + 0.025)
    )
    right_ok = right_ok_geometry & sample_right_known & sample_right_depth_ok

    # ========== 7. 应用填补 ==========
    use_right = hole_mask & right_ok

    gather_x = sample_x_smoothed.unsqueeze(-1).expand(h, w, 3)
    right_colour = torch.gather(image, 1, gather_x)
    result[hole_mask] = 0.0
    result[use_right] = right_colour[use_right]

    return result, sample_x, sample_x_smoothed


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

    # ========== 对比原版 vs 保守反遮挡带 ==========
    # 原版
    disocclusion_band_original = project_disocclusion_bands_conservative(
        disparity_sharp, min_drop=3.0, right_cleanup=16
    )
    hole_with_band_original = hole | disocclusion_band_original

    # v8 改进：更保守的
    disocclusion_band_conservative = project_disocclusion_bands_conservative(
        disparity_sharp, min_drop=args.min_drop, right_cleanup=args.band_cleanup
    )
    hole_with_band_conservative = hole | disocclusion_band_conservative

    # 目标空间 near
    target_near = b.forward_target_near(near_score, disparity_sharp, unreliable)
    b._LAST_TARGET_NEAR = target_near

    # ========== 原版 B 填补 ==========
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    final_right_b_original = b.strict_background_inpaint_gpu_b(
        right_warped.clone(), hole_with_band_original, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== v8 改进：保留镜像机制的平滑填补 ==========
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band_conservative] = 0.0

    final_right_v8, sample_x_raw, sample_x_smoothed = inpaint_with_smoothed_mirroring(
        right_with_band, hole_with_band_conservative, target_near,
        bg_threshold=0.3, safety_margin=6,
        smooth_kernel=args.smooth_kernel
    )

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'band_original': disocclusion_band_original.cpu().numpy(),
        'band_conservative': disocclusion_band_conservative.cpu().numpy(),
        'hole': hole.cpu().numpy(),
        'hole_original': hole_with_band_original.cpu().numpy(),
        'hole_conservative': hole_with_band_conservative.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final_b_original': final_right_b_original.cpu().numpy(),
        'final_v8': final_right_v8.cpu().numpy(),
        'sample_x_raw': sample_x_raw.cpu().numpy(),
        'sample_x_smoothed': sample_x_smoothed.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v8 保留几何的平滑填补 (kernel={args.smooth_kernel}, cleanup={args.band_cleanup}) ✨")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # 处理几个关键帧：第 60 帧(~2s), 第 90 帧(~3s), 第 210 帧(~7s), 第 240 帧(~8s)
    target_frames = [60, 90, 210, 240]

    for frame_idx in target_frames:
        cap = cv2.VideoCapture(args.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        cap.release()

        if not ok:
            print(f"无法读取第 {frame_idx} 帧")
            continue

        print(f"\n处理第 {frame_idx} 帧...")
        result = process_frame(frame_bgr, model, device, args)

        # 保存结果
        h, w = frame_bgr.shape[:2]

        # 1. B 版本原版结果
        final_b_u8 = (result['final_b_original'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v8_frame_{frame_idx:03d}_b_original.png'),
                   cv2.cvtColor(final_b_u8, cv2.COLOR_RGB2BGR))

        # 2. v8 改进结果
        final_v8_u8 = (result['final_v8'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v8_frame_{frame_idx:03d}_v8_improved.png'),
                   cv2.cvtColor(final_v8_u8, cv2.COLOR_RGB2BGR))

        # 3. 反遮挡带可视化对比
        viz = (result['warped'] * 255).astype(np.uint8)
        viz[result['band_original']] = [255, 0, 0]  # 红色：原版带
        viz[result['band_conservative']] = [0, 255, 0]  # 绿色：保守带
        cv2.imwrite(str(outdir / f'v8_frame_{frame_idx:03d}_band_compare.png'),
                   cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

        # 4. 空洞可视化对比
        hole_viz = (result['warped'] * 255).astype(np.uint8)
        hole_viz[result['hole_original']] = [255, 0, 0]  # 红色：原版空洞
        hole_viz[result['hole_conservative']] = [0, 255, 0]  # 绿色：保守空洞
        cv2.imwrite(str(outdir / f'v8_frame_{frame_idx:03d}_hole_compare.png'),
                   cv2.cvtColor(hole_viz, cv2.COLOR_RGB2BGR))

        # ========== 定量评估 ==========
        print(f"\n{'='*60}")
        print(f"📊 第 {frame_idx} 帧 评估结果")
        print(f"{'='*60}")

        # 反遮挡带面积对比
        area_original = result['band_original'].sum()
        area_conservative = result['band_conservative'].sum()
        print(f"反遮挡带像素数:")
        print(f"  原版: {area_original:,}")
        print(f"  保守: {area_conservative:,}")
        print(f"  减小: {100 * (1 - area_conservative / area_original):.1f}%")

        # 总空洞面积对比
        total_hole_original = result['hole_original'].sum()
        total_hole_conservative = result['hole_conservative'].sum()
        print(f"\n总空洞像素数:")
        print(f"  原版: {total_hole_original:,}")
        print(f"  保守: {total_hole_conservative:,}")
        print(f"  减小: {100 * (1 - total_hole_conservative / total_hole_original):.1f}%")

        # 取色位置平滑度（只在空洞区域统计）
        hole_mask = result['hole_conservative']
        if hole_mask.sum() > 0:
            sample_x_raw = result['sample_x_raw']
            sample_x_smoothed = result['sample_x_smoothed']

            # 统计垂直方向的变化
            grad_raw = np.abs(sample_x_raw[1:, :] - sample_x_raw[:-1, :])
            grad_smoothed = np.abs(sample_x_smoothed[1:, :] - sample_x_smoothed[:-1, :])

            # 只统计空洞区域
            hole_grad_mask = hole_mask[1:, :] & hole_mask[:-1, :]
            if hole_grad_mask.sum() > 0:
                print(f"\n取色位置垂直平滑度（空洞区域）:")
                print(f"  原版 - 平均: {grad_raw[hole_grad_mask].mean():.2f}px, 最大: {grad_raw[hole_grad_mask].max():.0f}px")
                print(f"  平滑 - 平均: {grad_smoothed[hole_grad_mask].mean():.2f}px, 最大: {grad_smoothed[hole_grad_mask].max():.0f}px")
                impr = (1 - grad_smoothed[hole_grad_mask].mean() / grad_raw[hole_grad_mask].mean()) * 100
                print(f"  改进: {impr:.1f}%")

        # 颜色平滑度对比
        y1, y2 = 0, h  # 全图分析
        x1, x2 = int(w * 0.4), int(w * 0.8)  # 关注人物右侧区域

        b_crop = result['final_b_original'][y1:y2, x1:x2]
        v8_crop = result['final_v8'][y1:y2, x1:x2]

        # 垂直颜色梯度
        b_grad_v = np.abs(b_crop[1:, :] - b_crop[:-1, :]).mean(axis=2)
        v8_grad_v = np.abs(v8_crop[1:, :] - v8_crop[:-1, :]).mean(axis=2)

        print(f"\n填补区域垂直颜色平滑度:")
        print(f"  B 原版 - 平均颜色差: {b_grad_v.mean():.4f}")
        print(f"  v8 改进 - 平均颜色差: {v8_grad_v.mean():.4f}")
        impr_color = (1 - v8_grad_v.mean() / max(b_grad_v.mean(), 1e-6)) * 100
        print(f"  颜色平滑改进: {impr_color:.1f}%")

    print(f"\n{'='*60}")
    print(f"✅ v8 迭代完成！所有结果保存在: {outdir}")
    print(f"   请对比查看:")
    print(f"   - *_b_original.png vs *_v8_improved.png: 最终效果")
    print(f"   - *_band_compare.png: 反遮挡带对比（红:原版, 绿:保守）")
    print(f"   - *_hole_compare.png: 总空洞对比（红:原版, 绿:保守）")


if __name__ == "__main__":
    main()
