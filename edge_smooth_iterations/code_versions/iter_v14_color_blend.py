"""
迭代 v14: 颜色空间的轻量垂直平滑 ✨

关键思想转变：
- 不要修改采样坐标（那样会扭曲背景几何结构，如天花板灯）
- 而是：先用原采样坐标填补（保证几何正确），然后对填补区域的颜色做轻微的垂直平滑

具体方案：
1. 完全使用 B 版本原算法做填补（保证天花板灯等几何结构正确）
2. 对填补区域左边界附近的颜色做 3-5 像素宽的垂直高斯平滑
3. 右边缘做羽化融合
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
    parser.add_argument("--left-smooth-width", type=int, default=6, help="左边界垂直平滑宽度")
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5, help="左边界平滑 sigma")
    parser.add_argument("--right-blend-width", type=int, default=4, help="右边缘羽化宽度")
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


# ========== v14: 颜色空间平滑 ==========
def inpaint_color_smoothing(image, hole_mask, target_near, near_score,
                             left_smooth_width=6, left_smooth_sigma=1.5,
                             right_blend_width=4):
    """
    ✨ 最佳方案：
    1. 先用 B 版本填补（保持背景几何结构，如天花板灯不会扭曲）
    2. 左边界附近：对颜色做窄带垂直高斯平滑（消除边缘锯齿）
    3. 右边界：羽化融合（消除接缝）
    """
    # ========== 步骤 1：完全使用 B 版本填补 ==========
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    result = b.strict_background_inpaint_gpu_b(
        image.clone(), hole_mask, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    h, w = hole_mask.shape
    device = hole_mask.device

    # ========== 步骤 2：左边界垂直颜色平滑 ==========
    if left_smooth_width > 0:
        # 找到每行空洞左边界
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        # 创建平滑掩码：左边界附近 left_smooth_width 像素
        smooth_mask = torch.zeros_like(hole_mask)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + left_smooth_width)
        smooth_mask = smooth_region & hole_mask

        # 对整个图像做垂直高斯模糊
        k = 5  # 核大小
        pad = k // 2
        sigma = left_smooth_sigma
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        # 只在平滑区域替换
        smooth_mask_3d = smooth_mask.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    # ========== 步骤 3：右边缘羽化融合 ==========
    if right_blend_width > 0:
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) == 0:
                continue

            right_edge_x = cols.max()
            if right_edge_x + 1 >= w:
                continue

            blend_start_x = max(cols.min(), right_edge_x - right_blend_width + 1)
            blend_x_range = torch.arange(blend_start_x, right_edge_x + 1, device=device)

            if len(blend_x_range) == 0:
                continue

            dist_from_edge = right_edge_x - blend_x_range
            alpha = (dist_from_edge.float() / right_blend_width).clamp(0.0, 1.0).view(-1, 1)

            inpainted_colors = result[y, blend_x_range]
            bg_color = result[y, right_edge_x + 1]
            blended = alpha * inpainted_colors + (1 - alpha) * bg_color
            result[y, blend_x_range] = blended

    return result


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

    # ========== v14 改进填补 ==========
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    final_right_v14 = inpaint_color_smoothing(
        right_with_band, hole_with_band, target_near, near_score,
        left_smooth_width=args.left_smooth_width,
        left_smooth_sigma=args.left_smooth_sigma,
        right_blend_width=args.right_blend_width
    )

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'band': disocclusion_band.cpu().numpy(),
        'hole': hole.cpu().numpy(),
        'hole_with_band': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final_b': final_right_b.cpu().numpy(),
        'final_v14': final_right_v14.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v14 颜色空间平滑 (left_width={args.left_smooth_width}, sigma={args.left_smooth_sigma}, right_blend={args.right_blend_width}) ✨")

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
        cv2.imwrite(str(outdir / f'v14_frame_{frame_idx:03d}_b_original.png'),
                   cv2.cvtColor(final_b_u8, cv2.COLOR_RGB2BGR))

        final_v14_u8 = (result['final_v14'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v14_frame_{frame_idx:03d}_v14_improved.png'),
                   cv2.cvtColor(final_v14_u8, cv2.COLOR_RGB2BGR))

        # ========== 定量评估 ==========
        print(f"\n{'='*60}")
        print(f"📊 第 {frame_idx} 帧 评估结果")
        print(f"{'='*60}")

        band_area = result['band'].sum()
        total_hole = result['hole_with_band'].sum()
        print(f"反遮挡带像素: {band_area:,}")
        print(f"总空洞像素: {total_hole:,}")

        # 填补-背景边界颜色差
        hole_mask = result['hole_with_band']
        b_result = result['final_b']
        v14_result = result['final_v14']

        boundary_diffs_b = []
        boundary_diffs_v14 = []
        for y in range(h):
            cols = np.where(hole_mask[y])[0]
            if len(cols) > 0:
                right_edge_x = cols.max()
                if right_edge_x + 1 < w:
                    diff_b = np.abs(b_result[y, right_edge_x] - b_result[y, right_edge_x + 1]).mean()
                    diff_v14 = np.abs(v14_result[y, right_edge_x] - v14_result[y, right_edge_x + 1]).mean()
                    boundary_diffs_b.append(diff_b)
                    boundary_diffs_v14.append(diff_v14)

        if len(boundary_diffs_b) > 0:
            print(f"\n填补-背景边界颜色差（右边缘）:")
            print(f"  B 原版 - 平均: {np.mean(boundary_diffs_b):.4f}, 最大: {np.max(boundary_diffs_b):.4f}")
            print(f"  v14 改进 - 平均: {np.mean(boundary_diffs_v14):.4f}, 最大: {np.max(boundary_diffs_v14):.4f}")
            impr = (1 - np.mean(boundary_diffs_v14) / np.mean(boundary_diffs_b)) * 100
            print(f"  边界改进: {impr:.1f}%")

        # 垂直颜色平滑度
        y1, y2 = 0, h
        x1, x2 = int(w * 0.4), int(w * 0.8)

        b_crop = b_result[y1:y2, x1:x2]
        v14_crop = v14_result[y1:y2, x1:x2]

        b_grad_v = np.abs(b_crop[1:, :] - b_crop[:-1, :]).mean(axis=2)
        v14_grad_v = np.abs(v14_crop[1:, :] - v14_crop[:-1, :]).mean(axis=2)

        print(f"\n填补区域垂直颜色平滑度:")
        print(f"  B 原版 - 平均颜色差: {b_grad_v.mean():.4f}")
        print(f"  v14 改进 - 平均颜色差: {v14_grad_v.mean():.4f}")
        impr_color = (1 - v14_grad_v.mean() / max(b_grad_v.mean(), 1e-6)) * 100
        print(f"  颜色平滑改进: {impr_color:.1f}%")

        # 只看左边界附近的垂直平滑
        hole_left_edge_diffs_b = []
        hole_left_edge_diffs_v14 = []
        for y in range(h - 1):
            cols_y = np.where(hole_mask[y])[0]
            cols_y1 = np.where(hole_mask[y + 1])[0]
            if len(cols_y) > 0 and len(cols_y1) > 0:
                left_x = min(cols_y.min(), cols_y1.min())
                for dx in range(0, min(15, len(cols_y), len(cols_y1))):
                    if left_x + dx < w - 1:
                        diff_b = np.abs(b_result[y, left_x + dx] - b_result[y + 1, left_x + dx]).mean()
                        diff_v14 = np.abs(v14_result[y, left_x + dx] - v14_result[y + 1, left_x + dx]).mean()
                        hole_left_edge_diffs_b.append(diff_b)
                        hole_left_edge_diffs_v14.append(diff_v14)

        if len(hole_left_edge_diffs_b) > 0:
            print(f"\n左边界附近 15 像素垂直平滑度:")
            print(f"  B 原版 - 平均: {np.mean(hole_left_edge_diffs_b):.4f}")
            print(f"  v14 改进 - 平均: {np.mean(hole_left_edge_diffs_v14):.4f}")
            impr_left = (1 - np.mean(hole_left_edge_diffs_v14) / np.mean(hole_left_edge_diffs_b)) * 100
            print(f"  左边界改进: {impr_left:.1f}%")

    print(f"\n{'='*60}")
    print(f"✅ v14 颜色空间平滑完成！结果保存在: {outdir}")
    print(f"   对比: *_b_original.png vs *_v14_improved.png")


if __name__ == "__main__":
    main()
