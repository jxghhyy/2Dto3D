"""
迭代 v24: 基于 B 版本 + 边缘后处理（推荐最终方案）✨

核心洞察：
- B 版本镜像采样 = 几何结构保持的天花板（天花板灯不扭曲）
- 左边界锯齿 + 接缝问题 = 可以通过后处理解决
- 大小核卷积 = 没必要，B 版本本身已经够好

最终方案：
1. 复用 B 版本的 strict_background_inpaint_gpu_b（镜像采样）
2. 左边界垂直高斯平滑（去锯齿）
3. 右边缘羽化（去接缝）
4. 保守反遮挡带（min_drop=3.5 + min_band_width=8）
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
    parser = argparse.ArgumentParser(description="v24: B版本 + 边缘后处理（最终推荐方案）")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--generate-side-by-side", action="store_true", help="生成并排对比（原版B vs v24）")

    # 锐化参数
    parser.add_argument("--sharpen-threshold", type=float, default=3.0, help="视差边缘锐化阈值")

    # 反遮挡带参数
    parser.add_argument("--band-min-drop", type=float, default=3.5, help="生成反遮挡带所需的视差下降量")
    parser.add_argument("--band-min-width", type=int, default=8, help="反遮挡带最小有效宽度")

    # 后处理参数
    parser.add_argument("--left-smooth-width", type=int, default=6, help="左边界垂直平滑宽度")
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5, help="左边界平滑 sigma")
    parser.add_argument("--right-blend-width", type=int, default=3, help="右边缘羽化宽度")

    return parser.parse_args()


def project_disocclusion_bands_optimized(disparity, min_drop=3.5, min_band_width=8):
    """优化的反遮挡带生成（参数可配置）"""
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    device = disparity.device
    right_cleanup = 10  # 默认值

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

    # 按宽度过滤小带（防止前景内部误判）
    band_np = band.cpu().numpy()
    for y in range(h):
        row = band_np[y]
        if row.sum() == 0:
            continue
        changes = np.diff(np.concatenate(([0], row.astype(int), [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < min_band_width:
                band_np[y, s:e] = False

    return torch.from_numpy(band_np).to(device)


def inpaint_optimized(image, hole_mask, near_score,
                      left_smooth_width=6, left_smooth_sigma=1.5, right_blend_width=3):
    """
    优化的填补流程：B版本 + 边缘后处理

    完全基于 B 版本的严格背景填补（镜像采样），
    只在后处理阶段做边缘平滑和羽化，不修改核心填补逻辑。
    """
    h, w = hole_mask.shape
    device = hole_mask.device

    # ========== 步骤1：B 版本严格背景填补（核心：镜像采样）==========
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    result = b.strict_background_inpaint_gpu_b(
        image.clone(), hole_mask, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== 步骤2：左边界垂直颜色平滑（消除锯齿）==========
    if left_smooth_width > 0:
        # 找到每行的空洞最左边界
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        # 只在边界附近做垂直平滑
        smooth_mask = torch.zeros_like(hole_mask)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + left_smooth_width)
        smooth_mask = smooth_region & hole_mask

        # 垂直高斯核
        k = 5
        pad = k // 2
        sigma = left_smooth_sigma
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        # 卷积（只在垂直方向）
        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        # 只替换平滑区域
        smooth_mask_3d = smooth_mask.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    # ========== 步骤3：右边缘羽化融合（消除填补-背景接缝）==========
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

            # 从右向左：alpha 从 0 到 1 渐变
            dist_from_edge = right_edge_x - blend_x_range
            alpha = (dist_from_edge.float() / right_blend_width).clamp(0.0, 1.0).view(-1, 1)

            inpainted_colors = result[y, blend_x_range]
            bg_color = result[y, right_edge_x + 1]  # 真实背景颜色
            blended = alpha * inpainted_colors + (1 - alpha) * bg_color
            result[y, blend_x_range] = blended

    return result


def process_frame(frame_bgr, model, device, args):
    """处理单帧"""
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
        disparity, kernel_size=15, threshold=args.sharpen_threshold,
        iterations=1, reject_margin=0.10
    )

    left_rgb = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    # ========== 优化的反遮挡带 ==========
    disocclusion_band = project_disocclusion_bands_optimized(
        disparity_sharp,
        min_drop=args.band_min_drop,
        min_band_width=args.band_min_width
    )
    hole_with_band = hole | disocclusion_band

    # ========== 原版 B 填补（用于对比）==========
    right_with_band_b = right_warped.clone()
    right_with_band_b[hole_with_band] = 0.0

    final_right_b = b.strict_background_inpaint_gpu_b(
        right_with_band_b, hole_with_band, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== v24 优化版填补（B版本 + 边缘后处理）==========
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    final_right_optimized = inpaint_optimized(
        right_with_band, hole_with_band, near_score,
        left_smooth_width=args.left_smooth_width,
        left_smooth_sigma=args.left_smooth_sigma,
        right_blend_width=args.right_blend_width
    )

    return final_right_b.cpu().numpy(), final_right_optimized.cpu().numpy()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v24 B版本 + 边缘后处理（最终推荐方案）✨")
    print(f"{'='*60}")
    print(f"  核心方案: B版本镜像采样（保留几何） + 边缘后处理（去锯齿/接缝）")
    print(f"  锐化阈值: {args.sharpen_threshold}")
    print(f"  反遮挡带: min_drop={args.band_min_drop}, min_width={args.band_min_width}")
    print(f"  左边界平滑: {args.left_smooth_width}px, sigma={args.left_smooth_sigma}")
    print(f"  右边缘羽化: {args.right_blend_width}px")
    print(f"{'='*60}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # 打开视频
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {total_frames} 帧, {fps} fps, {width}x{height}\n")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_opt = cv2.VideoWriter(str(outdir / 'v24_optimized_only.mp4'), fourcc, fps, (width, height))

    if args.generate_side_by_side:
        out_compare = cv2.VideoWriter(str(outdir / 'v24_side_by_side_compare.mp4'), fourcc, fps, (width * 2, height))

    from tqdm import tqdm
    for frame_idx in tqdm(range(total_frames), ncols=80):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        result_b, result_opt = process_frame(frame_bgr, model, device, args)

        # 转换为 BGR
        opt_u8 = (result_opt * 255).astype(np.uint8)
        opt_bgr = cv2.cvtColor(opt_u8, cv2.COLOR_RGB2BGR)
        out_opt.write(opt_bgr)

        if args.generate_side_by_side:
            b_u8 = (result_b * 255).astype(np.uint8)
            b_bgr = cv2.cvtColor(b_u8, cv2.COLOR_RGB2BGR)
            side_by_side = np.hstack([b_bgr, opt_bgr])
            out_compare.write(side_by_side)

    cap.release()
    out_opt.release()
    if args.generate_side_by_side:
        out_compare.release()

    print(f"\n{'='*60}")
    print(f"✅ v24 处理完成！结果保存在: {outdir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
