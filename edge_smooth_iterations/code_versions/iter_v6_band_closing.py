"""
迭代 v6: 反遮挡带的垂直闭运算（Vertical Morphological Closing）

最简单最直接的方法：
  反遮挡带生成后，做一个垂直方向的形态学闭运算
  → 先膨胀再腐蚀，填补垂直方向的小缺口
  → 边缘瞬间变平滑！

原理：如果上一行有反遮挡带，下一行没有，垂直膨胀会把它"连起来"，然后再腐蚀回去
结果：垂直方向小缺口被填补，边缘变平滑
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
    parser.add_argument("--closing-kernel", type=int, default=5, help="垂直闭运算核大小")
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


# ========== v6 改进：垂直闭运算 ==========
def project_disocclusion_bands_closing(disparity, min_drop=3.0, right_cleanup=16, closing_kernel=5):
    """
    生成反遮挡带后，做垂直方向的形态学闭运算（膨胀再腐蚀）
    消除垂直方向的"锯齿"，让边缘更平滑
    """
    band_raw = project_disocclusion_bands_original(disparity, min_drop, right_cleanup)

    h, w = band_raw.shape
    device = band_raw.device
    k = closing_kernel
    pad = k // 2

    # 转为 float
    band_float = band_raw.float().view(1, 1, h, w)

    # ========== 闭运算 = 先膨胀后腐蚀 ==========
    # 垂直方向膨胀：连接上下相邻行的缺口
    # 膨胀核：垂直方向 k 像素，水平方向 1 像素
    dilate_kernel = torch.ones((1, 1, k, 1), device=device, dtype=torch.float32)
    band_dilated = F.conv2d(band_float, dilate_kernel, padding=(pad, 0)) > 0.5

    # 垂直方向腐蚀：恢复原来的宽度
    erode_kernel = torch.ones((1, 1, k, 1), device=device, dtype=torch.float32)
    band_closed = F.conv2d(band_dilated.float(), erode_kernel, padding=(pad, 0)) > 0.5

    return band_closed[0, 0]


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

    # C版本：v6 改进 - 垂直闭运算后的反遮挡带
    disocclusion_band_raw = project_disocclusion_bands_original(
        disparity_sharp, min_drop=3.0, right_cleanup=16
    )
    disocclusion_band_closed = project_disocclusion_bands_closing(
        disparity_sharp, min_drop=3.0, right_cleanup=16, closing_kernel=args.closing_kernel
    )

    # 用闭运算后的带
    hole_with_band = hole | disocclusion_band_closed
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
        'band_raw': disocclusion_band_raw.cpu().numpy(),
        'band_closed': disocclusion_band_closed.cpu().numpy(),
        'hole_before': hole.cpu().numpy(),
        'hole_after': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final': final_right.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v6 反遮挡带垂直闭运算 (kernel={args.closing_kernel})")

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
    cv2.imwrite(str(outdir / 'v6_final_right.png'), cv2.cvtColor(final_u8, cv2.COLOR_RGB2BGR))

    # 可视化反遮挡带对比
    band_viz_raw = (result['warped'] * 255).astype(np.uint8)
    band_viz_raw[result['band_raw']] = [0, 255, 0]
    cv2.imwrite(str(outdir / 'v6_band_raw.png'), cv2.cvtColor(band_viz_raw, cv2.COLOR_RGB2BGR))

    band_viz_closed = (result['warped'] * 255).astype(np.uint8)
    band_viz_closed[result['band_closed']] = [0, 255, 0]
    cv2.imwrite(str(outdir / 'v6_band_closed.png'), cv2.cvtColor(band_viz_closed, cv2.COLOR_RGB2BGR))

    # 边缘裁剪
    y1, y2, x1, x2 = 300, 700, 900, 1100
    cv2.imwrite(str(outdir / 'v6_final_edge_crop.png'),
                cv2.cvtColor(final_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))

    # ========== 定量评估 ==========
    print("\n" + "="*60)
    print(f"📊 v6 反遮挡带垂直闭运算版本 评估")
    print("="*60)

    band_raw = result['band_raw']
    band_closed = result['band_closed']
    hole_after = result['hole_after']

    h_crop, w_crop = hole_after[y1:y2, x1:x2].shape
    hole_crop = hole_after[y1:y2, x1:x2]
    band_raw_crop = band_raw[y1:y2, x1:x2]
    band_closed_crop = band_closed[y1:y2, x1:x2]

    # 反遮挡带边缘平滑度对比
    edge_raw = np.zeros(h_crop, dtype=int)
    edge_closed = np.zeros(h_crop, dtype=int)
    for y in range(h_crop):
        cols_raw = np.where(band_raw_crop[y])[0]
        cols_closed = np.where(band_closed_crop[y])[0]
        if len(cols_raw) > 0:
            edge_raw[y] = cols_raw.min()
        if len(cols_closed) > 0:
            edge_closed[y] = cols_closed.min()

    valid_raw = edge_raw > 0
    valid_closed = edge_closed > 0

    if valid_raw.sum() > 0 and valid_closed.sum() > 0:
        grad_raw = np.abs(edge_raw[valid_raw][1:] - edge_raw[valid_raw][:-1])
        grad_closed = np.abs(edge_closed[valid_closed][1:] - edge_closed[valid_closed][:-1])

        print(f"反遮挡带左边界平滑度:")
        print(f"  原始 - 相邻行平均变化: {grad_raw.mean():.2f}, 最大: {grad_raw.max():.2f}")
        print(f"  闭运算 - 相邻行平均变化: {grad_closed.mean():.2f}, 最大: {grad_closed.max():.2f}")
        impr_mean = (1 - grad_closed.mean() / grad_raw.mean()) * 100
        impr_max = (1 - grad_closed.max() / grad_raw.max()) * 100
        print(f"  改进幅度 - 平均: {impr_mean:.1f}%, 最大: {impr_max:.1f}%  ← 越大越好！")

    # 填补后边缘颜色连续性
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

    print("\nv6 完成！结果保存到:", outdir)


if __name__ == "__main__":
    main()
