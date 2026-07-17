"""
迭代 v4: 前景边缘的垂直平滑

真正的问题：
  右视图中，左边是 warp 过来的前景（人物），右边是填补的背景
  前景边缘的颜色在垂直方向上可能有跳变（因为逐行 warp）
  虽然每行的颜色都对，但连起来看边缘是"锯齿状"的

方法：
  1. 找到前景边缘的位置（空洞的左边界）
  2. 提取边缘附近的颜色（前景侧 1 像素 + 背景侧 1 像素）
  3. 在垂直方向做平滑滤波
  4. 只平滑边缘颜色，不改变内部
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
    parser.add_argument("--edge-width", type=int, default=2, help="边缘平滑宽度")
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


# ========== v4 改进：前景边缘垂直平滑 ==========
def smooth_foreground_edge_vertical(image, hole_mask, smooth_kernel=5, edge_width=2):
    """
    平滑前景（左侧）和填补背景（右侧）之间的交界

    只平滑边缘附近的几行，内部颜色保持不变
    """
    h, w = hole_mask.shape
    device = hole_mask.device
    result = image.clone()

    # 找到每行空洞的左边界（这就是前景的右边缘）
    edge_x = torch.full((h,), w, device=device, dtype=torch.long)
    for y in range(h):
        cols = torch.where(hole_mask[y])[0]
        if len(cols) > 0:
            edge_x[y] = cols.min()  # 最左边的空洞像素 = 前景边缘位置

    # 提取边缘区域的颜色：前景侧 edge_width 像素 + 背景侧 edge_width 像素
    # 构建一个 2D 掩码标记需要平滑的区域
    smooth_mask = torch.zeros_like(hole_mask)
    x_range = torch.arange(w, device=device).view(1, w).expand(h, w)
    for offset in range(-edge_width, edge_width + 1):
        x_at_edge = edge_x.view(h, 1) + offset
        valid = (x_at_edge >= 0) & (x_at_edge < w)
        smooth_mask |= valid & (x_range == x_at_edge)

    # 对每个颜色通道做垂直方向平滑
    pad = smooth_kernel // 2
    sigma = smooth_kernel / 3.0
    kernel_1d = torch.exp(-(torch.arange(smooth_kernel, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d.view(1, 1, smooth_kernel, 1)  # [out, in, kH, kW]

    # 对整个图像做高斯模糊
    img_permuted = result.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    kernel_3ch = kernel_1d.view(1, 1, smooth_kernel, 1).repeat(3, 1, 1, 1)
    img_smoothed = F.conv2d(img_permuted, kernel_3ch,
                           padding=(pad, 0), groups=3)  # 垂直方向
    img_smoothed = img_smoothed[0].permute(1, 2, 0)  # [H, W, 3]

    # 只替换边缘区域
    smooth_mask_3d = smooth_mask.unsqueeze(-1)
    result = torch.where(smooth_mask_3d, img_smoothed, result)

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

    # C版本：反遮挡带
    disocclusion_band = project_disocclusion_bands_original(
        disparity_sharp, min_drop=3.0, right_cleanup=16
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

    final_right_raw = b.strict_background_inpaint_gpu_b(
        right_with_band, hole_with_band, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== v4 新增：前景边缘垂直平滑 ==========
    final_right_smooth = smooth_foreground_edge_vertical(
        final_right_raw, hole_with_band,
        smooth_kernel=args.smooth_kernel,
        edge_width=args.edge_width
    )

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'disocclusion_band': disocclusion_band.cpu().numpy(),
        'hole_before': hole.cpu().numpy(),
        'hole_after': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final_raw': final_right_raw.cpu().numpy(),
        'final_smooth': final_right_smooth.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v4 前景边缘垂直平滑 (kernel={args.smooth_kernel}, width={args.edge_width})")

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

    final_raw_u8 = (result['final_raw'] * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / 'v4_final_raw.png'), cv2.cvtColor(final_raw_u8, cv2.COLOR_RGB2BGR))

    final_smooth_u8 = (result['final_smooth'] * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / 'v4_final_smooth.png'), cv2.cvtColor(final_smooth_u8, cv2.COLOR_RGB2BGR))

    # 边缘裁剪
    y1, y2, x1, x2 = 300, 700, 900, 1100
    cv2.imwrite(str(outdir / 'v4_final_raw_edge_crop.png'),
                cv2.cvtColor(final_raw_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / 'v4_final_smooth_edge_crop.png'),
                cv2.cvtColor(final_smooth_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))

    # ========== 定量评估：边缘垂直平滑度 ==========
    print("\n" + "="*60)
    print(f"📊 v4 前景边缘垂直平滑评估")
    print("="*60)

    hole_after = result['hole_after']
    h_crop, w_crop = hole_after[y1:y2, x1:x2].shape
    hole_crop = hole_after[y1:y2, x1:x2]

    # 分析边缘位置的平滑度
    edge_x = np.zeros(h_crop, dtype=int)
    for y in range(h_crop):
        cols = np.where(hole_crop[y])[0]
        if len(cols) > 0:
            edge_x[y] = cols.min()

    valid = edge_x > 0
    edge_x_valid = edge_x[valid]

    if len(edge_x_valid) > 0:
        grad = np.abs(edge_x_valid[1:] - edge_x_valid[:-1])
        print(f"边缘位置平滑度 (空洞形状):")
        print(f"  相邻行平均变化: {grad.mean():.2f} 像素")
        print(f"  相邻行最大变化: {grad.max():.2f} 像素")

    # 分析边缘颜色的垂直平滑度
    raw_crop = result['final_raw'][y1:y2, x1:x2]
    smooth_crop = result['final_smooth'][y1:y2, x1:x2]

    edge_colors_raw = []
    edge_colors_smooth = []
    for y in range(h_crop):
        cols = np.where(hole_crop[y])[0]
        if len(cols) > 0 and cols.min() + 1 < w_crop:
            x = cols.min()  # 前景边缘位置
            edge_colors_raw.append(raw_crop[y, x])
            edge_colors_smooth.append(smooth_crop[y, x])

    if len(edge_colors_raw) > 1:
        edge_colors_raw = np.array(edge_colors_raw)
        edge_colors_smooth = np.array(edge_colors_smooth)

        # 计算相邻行的颜色差（垂直方向）
        color_grad_raw = np.abs(edge_colors_raw[1:] - edge_colors_raw[:-1]).mean(axis=1)
        color_grad_smooth = np.abs(edge_colors_smooth[1:] - edge_colors_smooth[:-1]).mean(axis=1)

        print(f"\n边缘颜色垂直平滑度:")
        print(f"  平滑前 - 平均颜色差: {color_grad_raw.mean():.3f}, 最大: {color_grad_raw.max():.3f}")
        print(f"  平滑后 - 平均颜色差: {color_grad_smooth.mean():.3f}, 最大: {color_grad_smooth.max():.3f}")
        improvement = (1 - color_grad_smooth.mean() / color_grad_raw.mean()) * 100
        print(f"  改进幅度: {improvement:.1f}%  ← 越大越好！")

    print("\nv4 完成！结果保存到:", outdir)


if __name__ == "__main__":
    main()
