"""
迭代 v3: 填补边缘羽化（Alpha Blending）

核心洞察：
  空洞形状是否平滑不重要！
  重要的是：填补区域和真实背景之间的颜色过渡是否平滑！

方法：
  1. 正常填补，得到填补结果
  2. 找到填补区域的右边缘（和真实背景交界）
  3. 对边缘 1-2 像素做 alpha 混合
  4. 垂直方向也考虑相邻行的颜色
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
    parser.add_argument("--blend-width", type=int, default=2, help="羽化宽度（像素）")
    parser.add_argument("--blend-sigma", type=float, default=0.8, help="羽化 sigma")
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


# ========== v3 改进：边缘羽化 ==========
def edge_blend_inpaint_result(inpainted_img, hole_mask, blend_width=2, blend_sigma=0.8):
    """
    对填补区域的边缘做羽化，消除硬边

    原理：
    填补区域的最右边几列，和右边的真实背景做 alpha 混合
    越靠近边缘，alpha 越低（背景权重越高）
    """
    h, w = hole_mask.shape
    device = hole_mask.device
    result = inpainted_img.clone()

    # 找到填补区域的右边缘（和真实背景的交界处）
    hole_float = hole_mask.float()
    # 右移一位找边缘：hole[i, x] == 1 且 hole[i, x+1] == 0
    hole_shifted = F.pad(hole_float[:, :-1], (1, 0))  # 左边补0，整体右移一位
    right_edge = hole_mask & (~(hole_shifted > 0.5))  # 当前是洞，左边是洞，右边不是 → 不对，应该反过来

    # 重新计算：当前是填补区域，右边是真实背景 → 这是填补区域的右边缘
    hole_shifted_left = F.pad(hole_float[:, 1:], (0, 1))  # 右边补0，整体左移一位
    right_edge = hole_mask & (~(hole_shifted_left > 0.5))  # 当前是洞，右边不是 → 右边缘

    # 计算每个空洞像素到右边缘的距离
    # 简单方法：从边缘向左 blend_width 像素都做混合
    edge_positions = torch.zeros((h, w), device=device, dtype=torch.long)
    for y in range(h):
        edge_cols = torch.where(right_edge[y])[0]
        if len(edge_cols) > 0:
            # 这一行最右边的边缘
            edge_pos = edge_cols.max()
            edge_positions[y, max(0, edge_pos - blend_width + 1):edge_pos + 1] = \
                torch.arange(blend_width, device=device).flip(0) + 1

    # 生成 alpha 权重（离边缘越近，alpha 越低 → 背景权重越高）
    alpha = torch.ones((h, w), device=device, dtype=torch.float32)
    blend_mask = edge_positions > 0
    alpha[blend_mask] = 1.0 - (edge_positions[blend_mask].float() / blend_width) * blend_sigma

    # 向右取背景颜色
    bg_colors = torch.zeros_like(result)
    bg_colors[:, :-1] = result[:, 1:]  # 右边相邻像素

    # alpha 混合
    alpha_3d = alpha.unsqueeze(-1)
    blend_region = hole_mask & blend_mask
    result[blend_region] = (alpha_3d[blend_region] * result[blend_region] +
                           (1 - alpha_3d[blend_region]) * bg_colors[blend_region])

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

    # C版本：反遮挡带（原版，不平滑）
    disocclusion_band = project_disocclusion_bands_original(
        disparity_sharp, min_drop=3.0, right_cleanup=16
    )
    hole_with_band = hole | disocclusion_band
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    # 目标空间 near
    target_near = b.forward_target_near(near_score, disparity_sharp, unreliable)
    b._LAST_TARGET_NEAR = target_near

    # 填补（原版）
    b._VARIANT_ARGS.strict_bg_safety_margin = 6
    b._VARIANT_ARGS.strict_bg_max_distance = 200
    b._VARIANT_ARGS.strict_bg_depth_tolerance = 0.025
    b._VARIANT_ARGS.narrow_hole_fallback_width = 10

    final_right_raw = b.strict_background_inpaint_gpu_b(
        right_with_band, hole_with_band, kernel_size=11, max_iter=64,
        stage_times={}, profile_sync=False,
        near=near_score, bg_threshold=0.3
    )

    # ========== v3 新增：边缘羽化 ==========
    final_right_blended = edge_blend_inpaint_result(
        final_right_raw, hole_with_band,
        blend_width=args.blend_width,
        blend_sigma=args.blend_sigma
    )

    return {
        'disparity': disparity_sharp.cpu().numpy(),
        'disocclusion_band': disocclusion_band.cpu().numpy(),
        'hole_before': hole.cpu().numpy(),
        'hole_after': hole_with_band.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'final_raw': final_right_raw.cpu().numpy(),
        'final_blended': final_right_blended.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 版本: v3 边缘羽化 (width={args.blend_width}, sigma={args.blend_sigma})")

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

    # 羽化前
    final_raw_u8 = (result['final_raw'] * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / 'v3_final_raw.png'), cv2.cvtColor(final_raw_u8, cv2.COLOR_RGB2BGR))

    # 羽化后
    final_blended_u8 = (result['final_blended'] * 255).astype(np.uint8)
    cv2.imwrite(str(outdir / 'v3_final_blended.png'), cv2.cvtColor(final_blended_u8, cv2.COLOR_RGB2BGR))

    # 边缘裁剪
    y1, y2, x1, x2 = 300, 700, 900, 1100
    cv2.imwrite(str(outdir / 'v3_final_raw_edge_crop.png'),
                cv2.cvtColor(final_raw_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / 'v3_final_blended_edge_crop.png'),
                cv2.cvtColor(final_blended_u8[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))

    # ========== 定量评估：边缘颜色连续性 ==========
    print("\n" + "="*60)
    print(f"📊 v3 边缘羽化版本 颜色连续性评估")
    print("="*60)

    hole_after = result['hole_after']
    h_crop, w_crop = hole_after[y1:y2, x1:x2].shape
    hole_crop = hole_after[y1:y2, x1:x2]

    # 分析填补区域右边缘的颜色差
    right_raw_crop = final_raw_u8[y1:y2, x1:x2].astype(np.float32) / 255.0
    right_blended_crop = final_blended_u8[y1:y2, x1:x2].astype(np.float32) / 255.0

    edge_color_diffs_raw = []
    edge_color_diffs_blended = []

    for y in range(h_crop):
        cols = np.where(hole_crop[y])[0]
        if len(cols) > 0:
            x = cols.max()  # 右边缘位置
            if x + 1 < w_crop:
                # 填补区域边缘 vs 右边背景的颜色差
                color_edge_raw = right_raw_crop[y, x]
                color_bg = right_raw_crop[y, x + 1]
                diff_raw = np.abs(color_edge_raw - color_bg).mean()
                edge_color_diffs_raw.append(diff_raw)

                color_edge_blended = right_blended_crop[y, x]
                diff_blended = np.abs(color_edge_blended - color_bg).mean()
                edge_color_diffs_blended.append(diff_blended)

    if len(edge_color_diffs_raw) > 0:
        print(f"填补边缘 → 背景 颜色差:")
        print(f"  羽化前（原版）: 平均={np.mean(edge_color_diffs_raw):.3f}, 最大={np.max(edge_color_diffs_raw):.3f}")
        print(f"  羽化后（改进）: 平均={np.mean(edge_color_diffs_blended):.3f}, 最大={np.max(edge_color_diffs_blended):.3f}")
        improvement = (1 - np.mean(edge_color_diffs_blended) / np.mean(edge_color_diffs_raw)) * 100
        print(f"  改进幅度: {improvement:.1f}%  ← 越大越好")

    print("\nv3 完成！结果保存到:", outdir)


if __name__ == "__main__":
    main()
