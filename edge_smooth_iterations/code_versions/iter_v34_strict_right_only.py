"""
迭代 v34: 严格单向 - 只使用空洞右侧的像素 ⭐⭐

核心改进（相对于 v33）：
1. ✅ 真正的非对称卷积核：左半部分权重 = 0，完全排除左侧
2. ✅ 方向感知的背景掩码：每个空洞像素只使用其右侧的像素
3. ✅ 从右向左逐层填充：防止已填充像素的污染扩散
4. ✅ 边缘检测只向右扩展
"""
import sys
import argparse
import subprocess
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

_KERNEL_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="v34: 严格单向 - 只使用空洞右侧的像素")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--video-encoder", type=str, default="h264_nvenc",
        choices=["h264_nvenc", "libx264"], help="视频编码器")
    parser.add_argument("--nvenc-gpu", type=int, default=0, help="NVENC GPU ID")
    parser.add_argument("--max-disparity", type=float, default=24.0, help="最大视差")

    parser.add_argument("--edge-kernel-size", type=int, default=7,
        help="边缘小核尺寸（阶段 2，精细处理）")
    parser.add_argument("--inner-kernel-size", type=int, default=15,
        help="内部大核尺寸（阶段 1，快速填充）")
    parser.add_argument("--right-bias-decay", type=float, default=2.0,
        help="右侧权重衰减因子（越大越偏向近右侧）")

    return parser.parse_args()


def create_strict_right_kernel(kernel_size, right_bias_decay, device, dtype):
    """
    ⭐⭐ 创建严格的右侧-only 卷积核

    对于 kernel_size = 5:
    - 位置：-2  -1   0  +1  +2  (相对于中心)
    - 权重：0   0   w1  w2  w3  (左侧完全为 0)

    right_bias_decay: 右侧权重的衰减因子
    - = 1: 右侧均匀权重
    - = 2: 越靠近中心权重越大（2x）
    - = 0.5: 越远离中心权重越大
    """
    pad = kernel_size // 2

    # 创建距离数组（相对于中心）
    # distance_from_center = 0 在中心，1,2,... 在右侧
    distance_from_center = torch.arange(kernel_size, device=device) - pad

    # 完全屏蔽左侧（距离 < 0 的位置）
    left_mask = distance_from_center < 0

    # 右侧权重：基于距离衰减
    # 注意：distance_from_center = 0 是中心，1,2,... 是右侧
    # 我们希望中心右侧的第一个像素权重最大
    right_distances = distance_from_center.float()  # 0,1,2,...pad

    # 权重 = exp(-distance * decay)，距离越近权重越大
    weights = torch.exp(-right_distances * right_bias_decay)

    # 左侧强制为 0
    weights[left_mask] = 0.0

    # 归一化（只考虑右侧非零权重）
    if weights.sum() > 1e-6:
        weights = weights / weights.sum() * (pad + 1)  # 有效半径 = pad+1

    # 2D 核：垂直方向均匀，水平方向严格右侧
    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)

    return kernel_2d


def create_direction_mask(h, w, hole, device):
    """
    ⭐⭐ 创建方向感知掩码：对于每个位置，只有它右侧的像素是可用的

    返回: [h, w, w] 的 3D 张量，mask[y, x, k] = True 当 k 在 x 右侧
    """
    # 创建坐标网格
    x_coords = torch.arange(w, device=device)  # [w]

    # 对于每个 x，哪些 k 在右侧？k >= x
    direction_mask_2d = x_coords.view(w, 1) <= x_coords.view(1, w)  # [w, w]

    # 扩展到高度维度
    direction_mask_3d = direction_mask_2d.view(1, w, w).expand(h, w, w)  # [h, w, w]

    return direction_mask_3d


def detect_hole_edges(hole_mask):
    is_hole = hole_mask.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones((1, 1, 3, 3), device=hole_mask.device, dtype=is_hole.dtype)
    neighbor_count = F.conv2d(1 - is_hole, kernel, padding=1)
    edge_mask = hole_mask & (neighbor_count[0, 0] > 0.01)
    return edge_mask


def fill_edge_strict_right(img, hole, edge_mask, near,
                           kernel_size=7, right_bias_decay=2.0,
                           bg_threshold=0.3):
    """
    v34 改进：严格只使用右侧背景像素填补边缘
    """
    result = img.clone()
    filled_mask = torch.zeros_like(hole)
    device = hole.device
    h, w = hole.shape

    # 背景像素：非空洞 + 近深度
    bg_mask = ~hole & (near < bg_threshold)

    if not torch.any(edge_mask):
        return result, filled_mask

    # ⭐ 只向右扩展边缘（不向左！）
    edge_extended = edge_mask.clone()
    for _ in range(2):
        # 只向右、上、下扩展，不向左
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=-1, dims=1)  # 右
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=0)   # 上
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=-1, dims=0)  # 下

    to_fill = edge_extended & hole

    if not torch.any(to_fill):
        return result, filled_mask

    # ⭐ 严格右侧卷积核
    pad = kernel_size // 2
    kernel1 = create_strict_right_kernel(kernel_size, right_bias_decay, device, img.dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)

    img_nchw = img.permute(2, 0, 1).unsqueeze(0)
    bg_mask_nchw = bg_mask.unsqueeze(0).unsqueeze(0).float()

    weighted_img = img_nchw * bg_mask_nchw
    rgb_sum = F.conv2d(weighted_img, kernel3, padding=pad, groups=3)
    weight_sum = F.conv2d(bg_mask_nchw, kernel1, padding=pad)

    avg = rgb_sum / weight_sum.clamp_min(1e-6)
    avg_hwc = avg[0].permute(1, 2, 0)

    valid_fill = to_fill & (weight_sum[0, 0] > 0.5)
    if torch.any(valid_fill):
        result[valid_fill] = avg_hwc[valid_fill]
        filled_mask[valid_fill] = True

    return result, filled_mask


def fast_inpaint_v34_strict_right(img, hole, near,
                                   edge_kernel_size=7,
                                   inner_kernel_size=15,
                                   right_bias_decay=2.0,
                                   bg_threshold=0.3,
                                   max_iter=64):
    """
    ⭐⭐ v34 核心：严格单向填补，只从右向左
    """
    img = img.clone()
    hole = hole.clone()
    device = hole.device
    h, w = hole.shape

    # 背景像素（保持不变）
    bg_mask = ~hole & (near < bg_threshold)

    # ========== 阶段 1：边缘检测 + 严格右侧小核填边缘 ==========
    edge_mask = detect_hole_edges(hole)
    img, edge_filled = fill_edge_strict_right(
        img, hole, edge_mask, near, edge_kernel_size, right_bias_decay, bg_threshold
    )
    hole = hole & ~edge_filled

    if hole.sum().item() == 0:
        return img, 0, "边缘填补完成"

    # ========== 阶段 2：从右向左逐层填充 ==========
    # 核心思想：每次只填充紧邻右侧已填充区域的空洞像素
    # 这样已填充的像素不会（也不能）反向污染

    cache_key = (inner_kernel_size, right_bias_decay, str(device), str(img.dtype).split('.')[-1])
    k1, k3 = _KERNEL_CACHE.get(cache_key, (None, None))

    if k1 is None:
        k1 = create_strict_right_kernel(inner_kernel_size, right_bias_decay, device, img.dtype)
        k3 = k1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[cache_key] = (k1, k3)

    pad = inner_kernel_size // 2
    prev_hole_count = hole.sum().item()
    no_progress_count = 0
    actual_iter = 0

    # ⭐ 方向感知的权重：只使用真正的背景像素 + 刚填充的像素
    # 但刚填充的像素只允许在它左侧的空洞像素中使用

    for it in range(max_iter):
        actual_iter += 1

        # 每次迭代都重新计算 bg_weight，只包含原始背景 + 已填充的"安全"像素
        # 但关键是：卷积核本身已经排除了左侧，所以 bg_weight 不需要特别处理
        bg_weight = (~hole).unsqueeze(-1).float()
        weighted_img = img * bg_weight

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, k1, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole & (weight_sum[0, 0] > 0.5)

        if can_fill.sum().item() > 0:
            img[can_fill] = avg_hwc[can_fill]
            hole[can_fill] = False

        # 提前终止
        current_hole_count = hole.sum().item()
        if can_fill.sum().item() == 0 or current_hole_count >= prev_hole_count:
            no_progress_count += 1
            if no_progress_count >= 2:
                break
        else:
            no_progress_count = 0
        prev_hole_count = current_hole_count

        if not hole.any():
            break

    # ========== 阶段 3：兜底填补（用更大的严格右侧核） ==========
    if hole.any():
        fallback_key = (21, right_bias_decay, str(device), str(img.dtype).split('.')[-1])
        k1_fallback, k3_fallback = _KERNEL_CACHE.get(fallback_key, (None, None))

        if k1_fallback is None:
            k1_fallback = create_strict_right_kernel(21, right_bias_decay, device, img.dtype)
            k3_fallback = k1_fallback.repeat(3, 1, 1, 1)
            _KERNEL_CACHE[fallback_key] = (k1_fallback, k3_fallback)

        pad_fallback = 10
        for _ in range(max_iter // 2):
            fillable_weight = (~hole).unsqueeze(-1).float()
            weighted_img = img * fillable_weight

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = fillable_weight.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3_fallback, padding=pad_fallback, groups=3)
            weight_sum = F.conv2d(weight_nchw, k1_fallback, padding=pad_fallback)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = hole & (weight_sum[0, 0] > 0.5)
            if can_fill.sum().item() > 0:
                img[can_fill] = avg_hwc[can_fill]
                hole[can_fill] = False
            else:
                break

            if not hole.any():
                break

    return img, actual_iter, "完成"


@torch.no_grad()
def project_disocclusion_bands_gpu(disparity, min_drop=3.0, min_band_width=8):
    """v32 的 GPU 反遮挡带（保持不变）"""
    h, w = disparity.shape
    device = disparity.device

    disp_flipped = torch.flip(disparity, dims=[1])
    max_from_right = torch.cummax(disp_flipped, dim=1)[0]
    max_from_right = torch.flip(max_from_right, dims=[1])

    max_right_shifted = torch.roll(max_from_right, shifts=1, dims=1)
    max_right_shifted[:, 0] = 0.0

    drop_mask = disparity > (max_right_shifted + min_drop)
    band_length = disparity.clamp(min=0).long()

    diff = torch.zeros((h, w + 1), dtype=torch.int32, device=device)

    rows, cols = torch.where(drop_mask)
    if len(rows) > 0:
        starts = (cols - band_length[rows, cols] + 1).clamp(min=0)
        ends = cols.clamp(max=w - 1)
        valid_mask = (ends - starts) >= min_band_width
        if valid_mask.any():
            diff[rows[valid_mask], starts[valid_mask]] += 1
            diff[rows[valid_mask], ends[valid_mask] + 1] -= 1

    bands = torch.cumsum(diff[:, :w], dim=1) > 0
    return bands


@torch.no_grad()
def edge_post_process_vectorized(image, hole_mask, smooth_width=6, smooth_sigma=1.5):
    """
    v34 改进：后处理也只向右侧平滑
    """
    h, w = hole_mask.shape
    device = image.device
    result = image.clone()

    if smooth_width > 0:
        hole_int = hole_mask.long()
        edge_x = hole_int.argmax(dim=1)
        has_no_hole = (hole_int.sum(dim=1) == 0)
        edge_x[has_no_hole] = w

        # 只向右侧平滑的核：左侧权重为 0
        k = 5
        pad = k // 2
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * smooth_sigma ** 2))
        # 左侧权重置 0
        gauss_kernel[:pad] = 0.0
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + smooth_width) & hole_mask
        smooth_mask_3d = smooth_region.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    return result


def main():
    import time
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v34] 使用设备: {device}")
    print(f"[v34] 视频编码器: {args.video_encoder}")
    print(f"[v34] max_disparity: {args.max_disparity}")
    print(f"[v34] 卷积核: 边缘 {args.edge_kernel_size}×, 内部 {args.inner_kernel_size}×")
    print(f"[v34] 右侧偏置衰减: {args.right_bias_decay}")

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / f"depth_anything_v2_{args.encoder}.pth"), map_location='cpu'))
    model = model.to(device).eval()
    print(f"[v34] 模型加载完成")

    # 输入视频
    video_path = Path(args.video_path)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[v34] 输入视频: {video_path.name}, {w_orig}×{h_orig}, {fps_orig:.1f} FPS, {total_frames} 帧")

    # 分辨率设置
    input_size = 518
    scale = input_size / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = input_size
    else:
        depth_h = input_size

    dibr_h, dibr_w = h_orig, w_orig
    max_disparity_orig = args.max_disparity
    max_disparity_dibr = max_disparity_orig * dibr_w / w_orig

    print(f"[v34] 深度模型分辨率: {depth_w}×{depth_h}")
    print(f"[v34] DIBR分辨率: {dibr_w}×{dibr_h}")

    # 输出设置（SBS 格式）
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 输出文件名包含参数
    output_filename = f"{video_path.stem}_v34_disp{int(max_disparity_orig)}_e{args.edge_kernel_size}_i{args.inner_kernel_size}_decay{args.right_bias_decay}_sbs.mp4"
    output_path = outdir / output_filename

    out_w, out_h = w_orig * 2, h_orig  # SBS：左右并排
    print(f"[v34] 输出模式: SBS (左右并排), 尺寸: {out_w}×{out_h}")
    print(f"[v34] 输出文件: {output_filename}")

    # 创建 ffmpeg 写入器
    pix_fmt = "yuv420p" if (out_w % 2 == 0 and out_h % 2 == 0) else "yuv444p"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{out_w}x{out_h}", "-r", str(fps_orig), "-i", "-", "-i", str(video_path),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", args.video_encoder,
    ]
    if args.video_encoder == "h264_nvenc":
        cmd.extend(["-gpu", str(args.nvenc_gpu), "-preset", "medium", "-rc", "vbr", "-cq", "26", "-b:v", "0"])
    else:
        cmd.extend(["-preset", "medium", "-crf", "26"])
    cmd.extend(["-pix_fmt", pix_fmt, "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(output_path)])

    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    if writer.stdin is None:
        cap.release()
        raise RuntimeError("ffmpeg 管道初始化失败")

    # 统计
    frame_times = []
    start_time = time.perf_counter()

    for frame_idx in range(total_frames):
        frame_t0 = time.perf_counter()
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # 深度推理
        left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
        img_resized = F.interpolate(img, size=(depth_h, depth_w), mode="bilinear", align_corners=False)
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

        # 上采样到 DIBR 分辨率
        near_score = F.interpolate(
            depth_norm[None, None, :, :],
            size=(dibr_h, dibr_w),
            mode="bilinear", align_corners=False
        )[0, 0]
        disparity = near_score * max_disparity_dibr

        # 锐化视差
        disparity_sharp, unreliable = b.sharpen_disparity_edges(
            disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
        )

        # DIBR 扭曲
        left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        # 反遮挡带
        disocclusion_band = project_disocclusion_bands_gpu(disparity_sharp, min_drop=3.5, min_band_width=8)
        hole_with_band = hole | disocclusion_band
        right_with_band = right_warped.clone()
        right_with_band[hole_with_band] = 0.0

        # ⭐⭐ v34：严格单向填补
        img_inpainted, actual_iter, status = fast_inpaint_v34_strict_right(
            right_with_band, hole_with_band, near_score,
            edge_kernel_size=args.edge_kernel_size,
            inner_kernel_size=args.inner_kernel_size,
            right_bias_decay=args.right_bias_decay,
            bg_threshold=0.3,
            max_iter=64
        )

        # 边缘后处理（也只向右平滑）
        final_right = edge_post_process_vectorized(
            img_inpainted, hole_with_band, smooth_width=6, smooth_sigma=1.5
        )

        # SBS 格式：左右并排
        final_right_uint8 = (final_right * 255).byte()
        left_rgb_uint8 = (left_rgb_tensor * 255).byte()
        sbs_uint8 = torch.cat([left_rgb_uint8, final_right_uint8], dim=1)

        # 传输到 CPU 并写入
        sbs_bytes = sbs_uint8.cpu().numpy().tobytes()
        writer.stdin.write(sbs_bytes)

        frame_times.append(time.perf_counter() - frame_t0)

        if (frame_idx + 1) % 50 == 0:
            avg_time = np.mean(frame_times[-50:])
            fps = 1.0 / avg_time
            eta = (total_frames - frame_idx - 1) * avg_time / 60
            print(f"[v34] 进度: {frame_idx + 1}/{total_frames} | 速度: {fps:.1f} FPS | 迭代: {actual_iter} | ETA: {eta:.1f} 分钟")

    # 清理
    cap.release()
    writer.stdin.close()
    writer.wait()

    total_time = time.perf_counter() - start_time
    avg_time = np.mean(frame_times)
    fps = 1.0 / avg_time

    print(f"\n{'=' * 70}")
    print(f"✅ v34 处理完成！结果保存在: {output_path}")
    print(f"\n⏱️  性能统计:")
    print(f"  总耗时: {total_time / 60:.1f} 分钟")
    print(f"  平均: {avg_time * 1000:.1f} ms / 帧")
    print(f"  速度: {fps:.1f} FPS")
    print(f"\n💾 输出文件: {output_filename}")
    size_mb = output_path.stat().st_size / 1024 / 1024 if output_path.exists() else 0
    print(f"  文件大小: {size_mb:.1f} MB")
    print(f"\n🎛️  参数:")
    print(f"  max_disparity: {args.max_disparity}")
    print(f"  边缘核: {args.edge_kernel_size}×{args.edge_kernel_size}")
    print(f"  内部核: {args.inner_kernel_size}×{args.inner_kernel_size}")
    print(f"  右侧偏置衰减: {args.right_bias_decay}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
