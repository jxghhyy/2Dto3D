"""
迭代 v20: 简单反遮挡带 + 前景保护裁剪 ✨

策略：
1. 使用标准 B 版本的反遮挡带生成（不做任何额外过滤）
2. 找到前景轮廓线
3. 只裁剪掉那些明显渗透到前景内部太深的带

目标：
- 左手和身体之间：不应该有反遮挡带（那是前景缝隙，不需要填补背景色）
- 天花板灯：正常保留反遮挡带
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
    parser.add_argument("--right-blend-width", type=int, default=3, help="右边缘羽化宽度")
    parser.add_argument("--band-cleanup", type=int, default=12, help="反遮挡带清理宽度")
    parser.add_argument("--min-drop", type=float, default=3.0, help="视差下降阈值")
    parser.add_argument("--allowed-penetration", type=int, default=20, help="允许反遮挡带渗透进前景的像素")
    return parser.parse_args()


# ========== 1. 标准 B 版本反遮挡带 + 前景裁剪 ==========
def project_disocclusion_bands_simple_protected(
    disparity, warp_valid_mask,
    min_drop=3.0, right_cleanup=12, allowed_penetration=20
):
    """
    最简单但最有效的方案：
    1. 完全像 B 版本那样生成反遮挡带（不做任何额外约束）
    2. 找到前景轮廓线
    3. 删除：在前景左边界左侧 > allowed_penetration 像素的带

    这样既保留正常的反遮挡带，又防止渗透到前景内部太深
    """
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool), np.zeros((h,), dtype=np.int64)

    device = disparity.device

    # ========== 步骤 A：找到每行前景的最右边界 ==========
    foreground_right_edge = torch.zeros((h,), dtype=torch.long, device=device)
    for y in range(h):
        valid_x = torch.where(warp_valid_mask[y])[0]
        if len(valid_x) > 0:
            foreground_right_edge[y] = valid_x.max()

    # ========== 步骤 B：标准 B 版本的反遮挡带生成 ==========
    x_left = torch.arange(w - 1, device=device).view(1, w - 1)
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

    difference = torch.zeros((h, w + 1), device=device, dtype=torch.int32)
    rows = torch.arange(h, device=device).view(h, 1).expand(h, w - 1)
    flat = difference.reshape(-1)
    start_index = (rows * (w + 1) + start)[valid]
    stop_index = (rows * (w + 1) + end + 1)[valid]
    flat.scatter_add_(0, start_index, torch.ones_like(start_index, dtype=flat.dtype))
    flat.scatter_add_(0, stop_index, -torch.ones_like(stop_index, dtype=flat.dtype))

    band = torch.cumsum(difference[:, :w], dim=1) > 0

    # ========== 步骤 C：✨ 简单但关键 - 删除前景内部太深的带 ✨ ==========
    x_full = torch.arange(w, device=device).view(1, w).expand(h, w)
    foreground_edge_full = foreground_right_edge.view(h, 1).expand(h, w)

    # 只删除那些在前景边界左侧超过 allowed_penetration 像素的带
    # 允许边界附近有一些带（因为深度估计有误差）
    too_far_inside_foreground = x_full < (foreground_edge_full - allowed_penetration)
    band = band & (~too_far_inside_foreground)

    return band, foreground_right_edge.cpu().numpy()


# ========== 2. 开运算移除小孤立孔洞 ==========
def clean_hole_mask_with_opening(hole_mask, kernel_size=2):
    h, w = hole_mask.shape
    device = hole_mask.device

    mask_float = hole_mask.float().view(1, 1, h, w)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=torch.float32)

    mask_eroded = F.conv2d(mask_float, kernel, padding=kernel_size//2) == kernel.sum()
    mask_opened = F.conv2d(mask_eroded.float(), kernel, padding=kernel_size//2) > 0

    return mask_opened[0, 0, :h, :w]


# ========== 3. 优化的填补算法 ==========
def inpaint_v20_optimized(image, hole_mask, target_near, near_score,
                          left_smooth_width=6, left_smooth_sigma=1.5,
                          right_blend_width=3):
    # 步骤 1：B 版本填补（保留几何结构）
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

    # 步骤 2：左边界垂直颜色平滑
    if left_smooth_width > 0:
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        smooth_mask = torch.zeros_like(hole_mask)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + left_smooth_width)
        smooth_mask = smooth_region & hole_mask

        k = 5
        pad = k // 2
        sigma = left_smooth_sigma
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask_3d = smooth_mask.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    # 步骤 3：右边缘羽化融合
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

    # warp 有效的区域 = ~hole
    warp_valid_mask = ~hole

    # ========== v20：简单保护的反遮挡带 ==========
    disocclusion_band, foreground_right_edge = project_disocclusion_bands_simple_protected(
        disparity_sharp, warp_valid_mask,
        min_drop=args.min_drop,
        right_cleanup=args.band_cleanup,
        allowed_penetration=args.allowed_penetration
    )
    hole_with_band = hole | disocclusion_band

    # 开运算移除小孤立孔洞
    hole_cleaned = clean_hole_mask_with_opening(hole_with_band, kernel_size=2)

    # 确保尺寸匹配
    h, w = right_warped.shape[:2]
    hole_cleaned = hole_cleaned[:h, :w]

    # 目标空间 near
    target_near = b.forward_target_near(near_score, disparity_sharp, unreliable)
    b._LAST_TARGET_NEAR = target_near

    # 优化版填补
    right_with_band = right_warped.clone()
    right_with_band[hole_cleaned] = 0.0

    final_right_optimized = inpaint_v20_optimized(
        right_with_band, hole_cleaned, target_near, near_score,
        left_smooth_width=args.left_smooth_width,
        left_smooth_sigma=args.left_smooth_sigma,
        right_blend_width=args.right_blend_width
    )

    return {
        'final_optimized': final_right_optimized.cpu().numpy(),
        'hole_original': hole_with_band.cpu().numpy(),
        'hole_cleaned': hole_cleaned.cpu().numpy(),
        'band': disocclusion_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'foreground_right_edge': foreground_right_edge,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v20 简单前景保护 ✨")
    print(f"参数: band_cleanup={args.band_cleanup}, min_drop={args.min_drop}, "
          f"allowed_penetration={args.allowed_penetration}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # 关键帧验证
    test_frames = [60, 90, 210, 240]
    print(f"\n处理关键帧验证: {test_frames}")

    for frame_idx in test_frames:
        cap = cv2.VideoCapture(args.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        cap.release()

        if not ok:
            continue

        print(f"\n处理帧 {frame_idx}...")
        result = process_frame(frame_bgr, model, device, args)

        # 保存
        h, w = frame_bgr.shape[:2]

        opt_u8 = (result['final_optimized'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v20_frame_{frame_idx:03d}_optimized.png'),
                   cv2.cvtColor(opt_u8, cv2.COLOR_RGB2BGR))

        # 反遮挡带可视化
        band_viz = (result['warped'] * 255).astype(np.uint8)
        band_viz[result['band']] = [0, 255, 0]
        cv2.imwrite(str(outdir / f'v20_frame_{frame_idx:03d}_band_green.png'),
                   cv2.cvtColor(band_viz, cv2.COLOR_RGB2BGR))

        # 画前景右边界线
        edge_viz = (result['warped'] * 255).astype(np.uint8)
        foreground_edge = result['foreground_right_edge']
        for y in range(h):
            if 0 <= foreground_edge[y] < w:
                cv2.circle(edge_viz, (int(foreground_edge[y]), y), 2, (255, 0, 0), -1)
        cv2.imwrite(str(outdir / f'v20_frame_{frame_idx:03d}_foreground_edge.png'),
                   cv2.cvtColor(edge_viz, cv2.COLOR_RGB2BGR))

        # 手部区域裁剪
        x1, x2 = 1000, 1300
        y1, y2 = 400, 700
        hand_crop = cv2.cvtColor(opt_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(outdir / f'v20_frame_{frame_idx:03d}_hand_crop.png'), hand_crop)

        # 统计
        band_pixels = result['band'].sum()
        orig_hole_pixels = result['hole_original'].sum()
        clean_hole_pixels = result['hole_cleaned'].sum()
        print(f"  反遮挡带像素: {band_pixels:,}")
        print(f"  原始空洞像素: {orig_hole_pixels:,}")
        print(f"  清理后空洞像素: {clean_hole_pixels:,}")

    print(f"\n✅ v20 关键帧处理完成！结果在: {outdir}")


if __name__ == "__main__":
    main()
