"""
迭代 v12: 仅边界区域的精确平滑 ✨

关键改进：
- 不再对整个 sample_x 场做滤波，只对空洞左边界附近的采样坐标做平滑
- 边界平滑后，保持内部镜像偏移的相对关系不变
- 只在边界处羽化，保持内部纹理清晰
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
    parser.add_argument("--smooth-kernel", type=int, default=5, help="边界垂直平滑核大小")
    parser.add_argument("--blend-width", type=int, default=4, help="边界羽化宽度")
    parser.add_argument("--band-cleanup", type=int, default=12, help="反遮挡带清理宽度")
    parser.add_argument("--min-drop", type=float, default=3.0, help="视差下降阈值")
    return parser.parse_args()


def project_disocclusion_bands_conservative(disparity, min_drop=3.0, right_cleanup=12):
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    x_left = torch.arange(w - 1, device=disparity.device).view(1, w - 1)
    d_left = disparity[:, :-1]
    d_right = disparity[:, 1:]

    is_drop = (d_left - d_right) >= min_drop
    disp_grad = torch.abs(d_left - d_right)
    grad_mean = disp_grad.mean()
    is_significant_edge = disp_grad >= (grad_mean * 0.3)
    is_drop = is_drop & is_significant_edge

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

    band = torch.cumsum(difference[:, :w], dim=1) > 0
    return band


# ========== v12: 仅边界精确平滑 ==========
def inpaint_boundary_only_smooth(image, hole_mask, target_near, bg_threshold=0.3,
                                 safety_margin=6, smooth_kernel=5, blend_width=4):
    """
    精确平滑方案：
    1. 先正常计算所有像素的采样坐标 sample_x
    2. 找到每行空洞左边界 edge_x[y]
    3. 只对边界位置的采样坐标做垂直平滑
    4. 计算每行偏移量 delta[y]，对整行空洞内所有采样坐标应用相同 delta
       → 保证内部相对偏移不变，纹理不扭曲
    5. 右边缘羽化融合
    """
    h, w = hole_mask.shape
    device = hole_mask.device
    result = image.clone()

    x = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
    known = (~hole_mask) & (target_near >= 0.0)

    # ========== B 版本流程 ==========
    right_bg = known.clone()
    for offset in range(1, safety_margin + 1):
        has_bg_left = torch.zeros_like(known)
        has_bg_left[:, offset:] = known[:, :-offset]
        right_bg &= has_bg_left

    right_seed = torch.where(right_bg, x, torch.full_like(x, w))
    right_index = torch.flip(
        torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values,
        dims=(1,),
    )

    # 左边界深度验证
    raw_left_seed = torch.where(known, x, torch.full_like(x, -1))
    raw_left_index = torch.cummax(raw_left_seed, dim=1).values
    left_boundary_exists = raw_left_index >= 0

    left_boundary_near = torch.gather(target_near, 1, raw_left_index.clamp(0, w - 1))
    right_boundary_near = torch.gather(target_near, 1, right_index.clamp(0, w - 1))
    right_depth_ok = (
        (~left_boundary_exists)
        | (right_boundary_near <= left_boundary_near + 0.025)
    )

    # 镜像偏移
    right_distance = right_index - x
    mirror_offset = (right_distance - safety_margin - 1).clamp_min(0)
    mirror_budget = (200 - right_distance).clamp_min(0)
    mirror_offset = torch.minimum(mirror_offset, mirror_budget)
    sample_x_original = (right_index + mirror_offset).clamp(0, w - 1)

    # ========== ✨ v12 核心改进：仅边界偏移平滑 ==========
    # 1. 找到每行空洞左边界
    edge_x = torch.full((h,), w, device=device, dtype=torch.long)
    for y in range(h):
        cols = torch.where(hole_mask[y])[0]
        if len(cols) > 0:
            edge_x[y] = cols.min()

    # 2. 提取边界位置的采样坐标
    valid_rows = edge_x < w
    if valid_rows.sum() > smooth_kernel:
        edge_sample_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            if edge_x[y] < w:
                edge_sample_x[y] = sample_x_original[y, edge_x[y]]

        # 3. 对边界采样坐标做垂直中值滤波
        k = smooth_kernel
        pad = k // 2
        edge_sample_x_4d = edge_sample_x.float().view(1, 1, h, 1)
        edge_sample_x_padded = F.pad(edge_sample_x_4d, (0, 0, pad, pad), mode='replicate')

        unfolded = F.unfold(edge_sample_x_padded, kernel_size=(k, 1), padding=0, stride=1)
        unfolded = unfolded.view(k, h)
        edge_sample_x_smoothed = torch.median(unfolded, dim=0)[0].round().long().clamp(0, w - 1)

        # 4. 计算每行的偏移量 delta
        delta = edge_sample_x_smoothed - edge_sample_x

        # 限制最大偏移（避免引入错误）
        max_delta = 4
        delta = delta.clamp(-max_delta, max_delta)

        # 5. 对整行空洞内所有采样坐标应用相同 delta
        #    关键：相对偏移完全不变 → 纹理不扭曲
        delta_2d = delta.view(h, 1).expand(h, w)
        sample_x_smoothed = (sample_x_original + delta_2d).clamp(0, w - 1)
    else:
        delta = torch.zeros_like(edge_x)
        sample_x_smoothed = sample_x_original

    # 深度验证
    sample_right_known = torch.gather(known, 1, sample_x_smoothed)
    sample_right_near = torch.gather(target_near, 1, sample_x_smoothed)
    sample_right_depth_ok = (
        (~left_boundary_exists)
        | (sample_right_near <= left_boundary_near + 0.025)
    )
    right_ok = right_depth_ok & sample_right_known

    # 应用填补
    use_right = hole_mask & right_ok
    gather_x = sample_x_smoothed.unsqueeze(-1).expand(h, w, 3)
    right_colour = torch.gather(image, 1, gather_x)
    result[hole_mask] = 0.0
    result[use_right] = right_colour[use_right]

    # ========== ✨ 右边缘羽化 ==========
    if blend_width > 0:
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) == 0:
                continue

            right_edge_x = cols.max()
            if right_edge_x + 1 >= w:
                continue

            blend_start_x = max(cols.min(), right_edge_x - blend_width + 1)
            blend_x_range = torch.arange(blend_start_x, right_edge_x + 1, device=device)

            if len(blend_x_range) == 0:
                continue

            dist_from_edge = right_edge_x - blend_x_range
            alpha = (dist_from_edge.float() / blend_width).clamp(0.0, 1.0).view(-1, 1)

            inpainted_colors = result[y, blend_x_range]
            bg_color = result[y, right_edge_x + 1]
            blended = alpha * inpainted_colors + (1 - alpha) * bg_color
            result[y, blend_x_range] = blended

    return result, sample_x_original, sample_x_smoothed, delta


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

    # 反遮挡带
    disocclusion_band = project_disocclusion_bands_conservative(
        disparity_sharp, min_drop=args.min_drop, right_cleanup=args.band_cleanup
    )
    hole_with_band = hole | disocclusion_band

    # 目标空间 near
    target_near = b.forward_target_near(near_score, disparity_sharp, unreliable)
    b._LAST_TARGET_NEAR = target_near

    # ========== 原版 B 填补 ==========
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    final_right_b = b.strict_background_inpaint_gpu_b(
        right_warped.clone(), hole_with_band, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== v12 改进填补 ==========
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    final_right_v12, sample_x_orig, sample_x_smoothed, delta = inpaint_boundary_only_smooth(
        right_with_band, hole_with_band, target_near,
        bg_threshold=0.3, safety_margin=6,
        smooth_kernel=args.smooth_kernel,
        blend_width=args.blend_width
    )

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'band': disocclusion_band.cpu().numpy(),
        'hole': hole.cpu().numpy(),
        'hole_with_band': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final_b': final_right_b.cpu().numpy(),
        'final_v12': final_right_v12.cpu().numpy(),
        'sample_x_orig': sample_x_orig.cpu().numpy(),
        'sample_x_smoothed': sample_x_smoothed.cpu().numpy(),
        'delta': delta.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v12 仅边界精确平滑 (smooth={args.smooth_kernel}, blend={args.blend_width}, cleanup={args.band_cleanup}) ✨")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # 处理关键帧
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

        final_b_u8 = (result['final_b'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v12_frame_{frame_idx:03d}_b_original.png'),
                   cv2.cvtColor(final_b_u8, cv2.COLOR_RGB2BGR))

        final_v12_u8 = (result['final_v12'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v12_frame_{frame_idx:03d}_v12_improved.png'),
                   cv2.cvtColor(final_v12_u8, cv2.COLOR_RGB2BGR))

        # ========== 定量评估 ==========
        print(f"\n{'='*60}")
        print(f"📊 第 {frame_idx} 帧 评估结果")
        print(f"{'='*60}")

        band_area = result['band'].sum()
        total_hole = result['hole_with_band'].sum()
        print(f"反遮挡带像素: {band_area:,}")
        print(f"总空洞像素: {total_hole:,}")

        # 偏移量统计
        delta = result['delta']
        valid_delta = np.abs(delta[np.abs(delta) < 100])
        if len(valid_delta) > 0:
            print(f"\n边界采样坐标偏移量:")
            print(f"  平均绝对值: {np.mean(np.abs(valid_delta)):.2f}px")
            print(f"  最大绝对值: {np.max(np.abs(valid_delta)):.0f}px")

        # 填补-背景边界颜色差
        hole_mask = result['hole_with_band']
        b_result = result['final_b']
        v12_result = result['final_v12']

        boundary_diffs_b = []
        boundary_diffs_v12 = []
        for y in range(h):
            cols = np.where(hole_mask[y])[0]
            if len(cols) > 0:
                right_edge_x = cols.max()
                if right_edge_x + 1 < w:
                    diff_b = np.abs(b_result[y, right_edge_x] - b_result[y, right_edge_x + 1]).mean()
                    diff_v12 = np.abs(v12_result[y, right_edge_x] - v12_result[y, right_edge_x + 1]).mean()
                    boundary_diffs_b.append(diff_b)
                    boundary_diffs_v12.append(diff_v12)

        if len(boundary_diffs_b) > 0:
            print(f"\n填补-背景边界颜色差（右边缘）:")
            print(f"  B 原版 - 平均: {np.mean(boundary_diffs_b):.4f}, 最大: {np.max(boundary_diffs_b):.4f}")
            print(f"  v12 改进 - 平均: {np.mean(boundary_diffs_v12):.4f}, 最大: {np.max(boundary_diffs_v12):.4f}")
            impr = (1 - np.mean(boundary_diffs_v12) / np.mean(boundary_diffs_b)) * 100
            print(f"  边界改进: {impr:.1f}%")

        # 垂直颜色平滑度
        y1, y2 = 0, h
        x1, x2 = int(w * 0.4), int(w * 0.8)

        b_crop = b_result[y1:y2, x1:x2]
        v12_crop = v12_result[y1:y2, x1:x2]

        b_grad_v = np.abs(b_crop[1:, :] - b_crop[:-1, :]).mean(axis=2)
        v12_grad_v = np.abs(v12_crop[1:, :] - v12_crop[:-1, :]).mean(axis=2)

        print(f"\n填补区域垂直颜色平滑度:")
        print(f"  B 原版 - 平均颜色差: {b_grad_v.mean():.4f}")
        print(f"  v12 改进 - 平均颜色差: {v12_grad_v.mean():.4f}")
        impr_color = (1 - v12_grad_v.mean() / max(b_grad_v.mean(), 1e-6)) * 100
        print(f"  颜色平滑改进: {impr_color:.1f}%")

    print(f"\n{'='*60}")
    print(f"✅ v12 仅边界精确平滑完成！结果保存在: {outdir}")
    print(f"   对比: *_b_original.png vs *_v12_improved.png")


if __name__ == "__main__":
    main()
