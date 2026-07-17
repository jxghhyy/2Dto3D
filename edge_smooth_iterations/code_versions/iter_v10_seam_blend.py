"""
迭代 v10: 填补-背景边界的羽化融合 ✨

重新分析问题：
1. 天花板灯扭曲 - 这是填补纹理和真实背景纹理不对齐导致的"接缝"
2. 手部扭曲 - 这是反遮挡带过度扩散，把前景边缘也挖掉了

v10 解决方案：
1. 保守的反遮挡带（减小 cleanup，提高阈值）- 保护手部细节
2. 在填补区域的**右边缘**（和真实背景交界处）做羽化融合
   - 不是垂直方向，而是水平方向！
   - 让填补纹理平滑过渡到真实背景
3. （可选）对填补区域做轻微的水平方向高斯模糊，减少硬边
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
    parser.add_argument("--blend-width", type=int, default=4, help="羽化宽度（像素）")
    parser.add_argument("--band-cleanup", type=int, default=12, help="反遮挡带清理宽度")
    parser.add_argument("--min-drop", type=float, default=3.0, help="视差下降阈值")
    return parser.parse_args()


# ========== 保守的反遮挡带（更精细的控制） ==========
def project_disocclusion_bands_optimized(disparity, min_drop=3.0, right_cleanup=12):
    """
    优化的反遮挡带：
    1. 对视差图做轻微的平滑，减少噪声导致的误判
    2. 只在真正的深度不连续处生成带
    """
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    # 1. 先对视差图做轻微的垂直方向平滑，减少噪声
    disp_smoothed = disparity.clone()
    k = 3
    pad = 1
    disp_4d = disp_smoothed.view(1, 1, h, w)
    disp_padded = F.pad(disp_4d, (0, 0, pad, pad), mode='replicate')
    kernel = torch.ones((1, 1, k, 1), device=disparity.device, dtype=torch.float32) / k
    disp_smoothed = F.conv2d(disp_padded, kernel)[0, 0]

    x_left = torch.arange(w - 1, device=disparity.device).view(1, w - 1)
    d_left = disp_smoothed[:, :-1]
    d_right = disp_smoothed[:, 1:]

    # 2. 视差下降必须同时满足：
    #    a. 绝对阈值 min_drop
    #    b. 相对阈值（相对于局部梯度）
    disp_grad = torch.abs(d_left - d_right)
    grad_mean = disp_grad.mean()
    is_drop = (d_left - d_right) >= min_drop
    is_large_grad = disp_grad >= (grad_mean * 0.5)
    is_drop = is_drop & is_large_grad

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


# ========== v10 核心：填补-背景边界的水平羽化 ==========
def blend_inpaint_boundary(image, hole_mask, blend_width=4):
    """
    ✨ 核心：填补区域右边缘和真实背景的水平羽化

    问题：填补区域最右边的像素和右边真实背景之间有硬边
    解决：对填补区域右边缘的 blend_width 个像素做 alpha 混合

    算法：
    1. 找到每行填补区域的右边界 x
    2. 从边界向左 blend_width 像素，计算 alpha 权重
    3. alpha = x_dist / blend_width  (越靠近边界，背景权重越高)
    4. result = alpha * inpainted + (1 - alpha) * background
    """
    h, w = hole_mask.shape
    device = hole_mask.device
    result = image.clone()

    # 对每行处理
    for y in range(h):
        cols = torch.where(hole_mask[y])[0]
        if len(cols) == 0:
            continue

        # 这一行填补区域的右边界
        right_edge_x = cols.max()

        # 如果边界在图像边缘，跳过
        if right_edge_x + 1 >= w:
            continue

        # 需要羽化的区域：从右边界向左 blend_width 像素
        blend_start_x = max(cols.min(), right_edge_x - blend_width + 1)
        blend_x_range = torch.arange(blend_start_x, right_edge_x + 1, device=device)

        if len(blend_x_range) == 0:
            continue

        # 计算 alpha 权重：从左到右 alpha 从 1 降到 0
        # x_dist = right_edge_x - x (x 越靠右，dist 越小，alpha 越小)
        x_dist = right_edge_x - blend_x_range
        alpha = (x_dist.float() / blend_width).clamp(0.0, 1.0)
        alpha = alpha.view(-1, 1)  # 每个 x 位置一个 alpha

        # 填补颜色和右边背景颜色
        inpainted_colors = result[y, blend_x_range]  # [N, 3]
        bg_color = result[y, right_edge_x + 1]  # [3] 右边真实背景

        # 混合：越靠近边界，背景权重越大
        blended = alpha * inpainted_colors + (1 - alpha) * bg_color

        # 应用
        result[y, blend_x_range] = blended

    return result


# ========== v10 额外：填补区域轻微柔化 ==========
def soft_inpaint_region(image, hole_mask, sigma=0.5):
    """
    对填补区域做轻微的高斯模糊，减少纹理噪声
    只在空洞区域内做，不影响前景和真实背景
    """
    if hole_mask.sum() == 0:
        return image

    result = image.clone()

    # 对整个图像做高斯模糊
    k = 3
    pad = 1
    img_4d = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    img_padded = F.pad(img_4d, (pad, pad, pad, pad), mode='reflect')

    # 高斯核
    x = torch.arange(k, device=image.device, dtype=torch.float32) - k // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.view(k, 1) @ gauss.view(1, k)
    kernel = kernel_2d.view(1, 1, k, k).repeat(3, 1, 1, 1)

    img_blurred = F.conv2d(img_padded, kernel, groups=3)[0].permute(1, 2, 0)

    # 只在空洞区域用模糊结果
    result[hole_mask] = img_blurred[hole_mask]

    return result


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

    # ========== 反遮挡带 ==========
    disocclusion_band = project_disocclusion_bands_optimized(
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

    # ========== v10 改进：边界羽化融合 ==========
    # 先做轻微柔化，再做边界羽化
    final_right_v10 = soft_inpaint_region(final_right_b.clone(), hole_with_band, sigma=0.5)
    final_right_v10 = blend_inpaint_boundary(final_right_v10, hole_with_band, blend_width=args.blend_width)

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'band': disocclusion_band.cpu().numpy(),
        'hole': hole.cpu().numpy(),
        'hole_with_band': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final_b': final_right_b.cpu().numpy(),
        'final_v10': final_right_v10.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v10 填补-背景边界羽化 (blend={args.blend_width}, cleanup={args.band_cleanup}) ✨")

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

        # 1. B 版本原版
        final_b_u8 = (result['final_b'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v10_frame_{frame_idx:03d}_b_original.png'),
                   cv2.cvtColor(final_b_u8, cv2.COLOR_RGB2BGR))

        # 2. v10 改进
        final_v10_u8 = (result['final_v10'] * 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f'v10_frame_{frame_idx:03d}_v10_improved.png'),
                   cv2.cvtColor(final_v10_u8, cv2.COLOR_RGB2BGR))

        # 3. 反遮挡带可视化
        band_viz = (result['warped'] * 255).astype(np.uint8)
        band_viz[result['band']] = [0, 255, 0]
        cv2.imwrite(str(outdir / f'v10_frame_{frame_idx:03d}_band.png'),
                   cv2.cvtColor(band_viz, cv2.COLOR_RGB2BGR))

        # 4. 空洞可视化
        hole_viz = (result['warped'] * 255).astype(np.uint8)
        hole_viz[result['hole_with_band']] = [0, 255, 0]
        cv2.imwrite(str(outdir / f'v10_frame_{frame_idx:03d}_hole.png'),
                   cv2.cvtColor(hole_viz, cv2.COLOR_RGB2BGR))

        # ========== 定量评估 ==========
        print(f"\n{'='*60}")
        print(f"📊 第 {frame_idx} 帧 评估结果")
        print(f"{'='*60}")

        # 反遮挡带面积
        band_area = result['band'].sum()
        total_hole = result['hole_with_band'].sum()
        print(f"反遮挡带像素: {band_area:,}")
        print(f"总空洞像素: {total_hole:,}")

        # 边界颜色差（填补和背景的交界处）
        hole_mask = result['hole_with_band']
        b_result = result['final_b']
        v10_result = result['final_v10']

        boundary_color_diffs_b = []
        boundary_color_diffs_v10 = []
        for y in range(h):
            cols = np.where(hole_mask[y])[0]
            if len(cols) > 0:
                right_edge_x = cols.max()
                if right_edge_x + 1 < w:
                    inpainted_color_b = b_result[y, right_edge_x]
                    inpainted_color_v10 = v10_result[y, right_edge_x]
                    bg_color = b_result[y, right_edge_x + 1]

                    diff_b = np.abs(inpainted_color_b - bg_color).mean()
                    diff_v10 = np.abs(inpainted_color_v10 - bg_color).mean()

                    boundary_color_diffs_b.append(diff_b)
                    boundary_color_diffs_v10.append(diff_v10)

        if len(boundary_color_diffs_b) > 0:
            print(f"\n填补-背景边界颜色差（右边缘）:")
            print(f"  B 原版 - 平均: {np.mean(boundary_color_diffs_b):.4f}, 最大: {np.max(boundary_color_diffs_b):.4f}")
            print(f"  v10 改进 - 平均: {np.mean(boundary_color_diffs_v10):.4f}, 最大: {np.max(boundary_color_diffs_v10):.4f}")
            impr = (1 - np.mean(boundary_color_diffs_v10) / np.mean(boundary_color_diffs_b)) * 100
            print(f"  边界改进: {impr:.1f}%")

        # 整体垂直颜色平滑度
        y1, y2 = 0, h
        x1, x2 = int(w * 0.4), int(w * 0.8)

        b_crop = b_result[y1:y2, x1:x2]
        v10_crop = v10_result[y1:y2, x1:x2]

        b_grad_v = np.abs(b_crop[1:, :] - b_crop[:-1, :]).mean(axis=2)
        v10_grad_v = np.abs(v10_crop[1:, :] - v10_crop[:-1, :]).mean(axis=2)

        print(f"\n填补区域垂直颜色平滑度:")
        print(f"  B 原版 - 平均颜色差: {b_grad_v.mean():.4f}")
        print(f"  v10 改进 - 平均颜色差: {v10_grad_v.mean():.4f}")
        impr_color = (1 - v10_grad_v.mean() / max(b_grad_v.mean(), 1e-6)) * 100
        print(f"  颜色平滑改进: {impr_color:.1f}%")

    print(f"\n{'='*60}")
    print(f"✅ v10 迭代完成！结果保存在: {outdir}")
    print(f"   对比: *_b_original.png vs *_v10_improved.png")


if __name__ == "__main__":
    main()
