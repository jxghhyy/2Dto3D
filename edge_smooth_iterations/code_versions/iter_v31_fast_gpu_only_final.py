"""
迭代 v31: 极速版 - 仅输出最终 SBS 视频 + GPU ffmpeg 编码 ⚡

核心改进：
1. ✅ 仅输出最终立体视频（移除 9 个中间阶段视频，大幅提速）
2. ✅ GPU ffmpeg 编码（h264_nvenc），和 mono2stereo.py 一致
3. ✅ 自动适配：如果 sbs 宽度超标，自动切换到 ou 模式
4. ✅ 保留 v30 的所有核心算法（提前终止、大小核、兜底填补）
"""
import sys
import argparse
import subprocess
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# 获取脚本所在目录，构建绝对路径
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "submodules/depth/dav2"))
from depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# 全局核缓存
_KERNEL_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="v31: 极速版 - 仅输出最终视频 + GPU编码")
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")

    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--video-encoder", type=str, default="h264_nvenc",
        choices=["h264_nvenc", "libx264"], help="视频编码器")
    parser.add_argument("--nvenc-gpu", type=int, default=0, help="NVENC GPU ID")
    parser.add_argument("--nvenc-preset", type=str, default="medium", help="NVENC preset")
    parser.add_argument("--nvenc-cq", type=int, default=26, help="NVENC CQ质量")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "sbs", "ou"], help="输出布局")

    parser.add_argument("--sharpen-threshold", type=float, default=3.0)
    parser.add_argument("--band-min-drop", type=float, default=3.5)
    parser.add_argument("--band-min-width", type=int, default=8)

    parser.add_argument("--edge-kernel-size", type=int, default=5, help="边缘小核")
    parser.add_argument("--non-edge-kernel-size", type=int, default=11, help="内部大核")
    parser.add_argument("--max-iter", type=int, default=64, help="最大迭代（有提前终止）")

    parser.add_argument("--left-smooth-width", type=int, default=6, help="左边界垂直平滑宽度")
    parser.add_argument("--left-smooth-sigma", type=float, default=1.5, help="左边界平滑 sigma")
    parser.add_argument("--right-blend-width", type=int, default=3, help="右边缘羽化宽度")

    return parser.parse_args()


def _get_inpaint_kernels(kernel_size, device, dtype_str):
    key = (kernel_size, str(device), dtype_str)
    if key not in _KERNEL_CACHE:
        kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=getattr(torch, dtype_str))
        kernel3 = kernel1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[key] = (kernel1, kernel3)
    return _KERNEL_CACHE[key]


def detect_hole_edges(hole_mask):
    is_hole = hole_mask.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones((1, 1, 3, 3), device=hole_mask.device, dtype=is_hole.dtype)
    neighbor_count = F.conv2d(1 - is_hole, kernel, padding=1)
    edge_mask = hole_mask & (neighbor_count[0, 0] > 0.01)
    return edge_mask


def fill_edge_with_nearest_bg(img, hole, edge_mask, near, bg_threshold=0.3):
    result = img.clone()
    filled_mask = torch.zeros_like(hole)
    h, w = hole.shape
    device = hole.device

    bg_mask = ~hole & (near < bg_threshold)

    if torch.any(edge_mask):
        edge_extended = edge_mask.clone()
        for _ in range(2):
            edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=1) | torch.roll(edge_extended, shifts=-1, dims=1)
            edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=0) | torch.roll(edge_extended, shifts=-1, dims=0)

        to_fill = edge_extended & hole

        if torch.any(to_fill):
            img_nchw = img.permute(2, 0, 1).unsqueeze(0)
            bg_mask_nchw = bg_mask.unsqueeze(0).unsqueeze(0).float()

            pad = 2
            kernel1 = torch.ones((1, 1, 5, 5), device=device, dtype=img.dtype)
            kernel3 = kernel1.repeat(3, 1, 1, 1)

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


def fast_inpaint_v30(img, hole, near, bg_threshold=0.3, edge_kernel_size=5, non_edge_kernel_size=11, max_iter=64):
    """v30: 带提前终止的边缘敏感大小核填补"""
    img = img.clone()
    hole = hole.clone()
    h, w = hole.shape
    device = hole.device
    dtype = img.dtype

    # 边缘像素直接用周围背景填补（小核或最近邻）
    edge_mask = detect_hole_edges(hole)

    # 模式0: 边缘用小核，内部用大核
    img, edge_filled = fill_edge_with_nearest_bg(img, hole, edge_mask, near, bg_threshold)
    hole = hole & ~edge_filled

    hole_pixels_remaining = hole.sum().item()
    if hole_pixels_remaining == 0:
        return img, 0, "边缘填补完成"

    # 内部用大核
    k1, k3 = _get_inpaint_kernels(non_edge_kernel_size, device, str(dtype).split('.')[-1])
    pad = non_edge_kernel_size // 2

    # 开始迭代 - 带提前终止
    prev_hole_count = hole.sum().item()
    no_progress_count = 0

    for it in range(max_iter):
        # 构建权重：只有背景像素才有权重
        bg_weight = (~hole).unsqueeze(-1).float()
        weighted_img = img * bg_weight

        # 加权求和
        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, k1, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        # 更新空洞位置
        can_fill = hole & (weight_sum[0, 0] > 0.5)
        filled_this_iter = can_fill.sum().item()

        if filled_this_iter > 0:
            img[can_fill] = avg_hwc[can_fill]
            hole[can_fill] = False

        # 提前终止检测
        current_hole_count = hole.sum().item()
        if filled_this_iter == 0 or current_hole_count >= prev_hole_count:
            no_progress_count += 1
            if no_progress_count >= 2:
                break
        else:
            no_progress_count = 0
        prev_hole_count = current_hole_count

        if not hole.any():
            break

    # 如果还有空洞，进入兜底填补（放宽背景限制）
    status = "正常填补完成"
    actual_iter = it + 1

    if hole.any():
        status = "进入兜底填补"
        hole_pixels_before_fallback = hole.sum().item()
        fallback_iter = 0

        k1_fallback, k3_fallback = _get_inpaint_kernels(15, device, str(dtype).split('.')[-1])
        pad_fallback = 7

        for _ in range(max_iter // 2):
            fallback_iter += 1
            fillable_weight = (~hole).unsqueeze(-1).float()
            weighted_img = img * fillable_weight

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = fillable_weight.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3_fallback, padding=pad_fallback, groups=3)
            weight_sum = F.conv2d(weight_nchw, k1_fallback, padding=pad_fallback)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = hole & (weight_sum[0, 0] > 0.5)
            filled_this_fallback = can_fill.sum().item()

            if filled_this_fallback > 0:
                img[can_fill] = avg_hwc[can_fill]
                hole[can_fill] = False
            else:
                break

            if not hole.any():
                break

        actual_iter += fallback_iter

    # 最终兜底：如果还有空洞，用全局均值
    if hole.any():
        status = "使用全局均值兜底"
        mean_color = img[~hole].mean(dim=0)
        img[hole] = mean_color

    return img, actual_iter, status


@torch.no_grad()
def project_disocclusion_bands_gpu(disparity, min_drop=3.0, min_band_width=8):
    """GPU 加速版：投影反遮挡带（全向量化，无循环）"""
    h, w = disparity.shape
    device = disparity.device

    # 1. 从右向左的 max pooling（用 flip + cummax 实现）
    disp_flipped = torch.flip(disparity, dims=[1])
    max_from_right = torch.cummax(disp_flipped, dim=1)[0]
    max_from_right = torch.flip(max_from_right, dims=[1])

    # 2. 探测视差下降点（当前视差 > 右侧max视差 + min_drop）
    # 右侧max视差向右移1位（用roll，最右列设为0）
    max_right_shifted = torch.roll(max_from_right, shifts=1, dims=1)
    max_right_shifted[:, 0] = 0.0

    # 找出需要投影带的列
    drop_mask = disparity > (max_right_shifted + min_drop)

    # 3. 计算每列需要投影的带长度
    band_length = disparity.clamp(min=0).long()  # d 像素

    # 4. 用 difference array 算法填充投影区域
    # 创建 difference array（w+1 防止越界）
    diff = torch.zeros((h, w + 1), dtype=torch.int32, device=device)

    # 获取所有需要投影的坐标
    rows, cols = torch.where(drop_mask)
    if len(rows) > 0:
        band_lengths = band_length[rows, cols]
        starts = (cols - band_lengths + 1).clamp(min=0)
        ends = cols.clamp(max=w - 1)

        # 只保留满足最小宽度的带
        valid_mask = (ends - starts) >= min_band_width
        if valid_mask.any():
            rows_valid = rows[valid_mask]
            starts_valid = starts[valid_mask]
            ends_valid = ends[valid_mask]

            # difference array 更新
            diff[rows_valid, starts_valid] += 1
            diff[rows_valid, ends_valid + 1] -= 1

    # 5. cumsum 得到带
    bands = torch.cumsum(diff[:, :w], dim=1) > 0

    return bands


@torch.no_grad()
def edge_post_process(image, hole_mask, smooth_width=6, smooth_sigma=1.5, blend_width=3):
    """左边界垂直平滑 + 右边缘羽化"""
    h, w = hole_mask.shape
    device = image.device

    result = image.clone()

    # 左边界垂直平滑
    if smooth_width > 0:
        edge_x = torch.full((h,), w, device=device, dtype=torch.long)
        for y in range(h):
            cols = torch.where(hole_mask[y])[0]
            if len(cols) > 0:
                edge_x[y] = cols.min()

        k = 5
        pad = k // 2
        sigma = smooth_sigma
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        smooth_mask = torch.zeros_like(hole_mask)
        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + smooth_width) & hole_mask
        smooth_mask_3d = smooth_region.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    # 右边缘羽化
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

    return result


def create_nvenc_writer(ffmpeg_bin, input_video, output_video, fps, out_w, out_h, encoder, nvenc_gpu, preset, cq):
    """创建 ffmpeg GPU 编码写入器"""
    pix_fmt = "yuv420p" if (out_w % 2 == 0 and out_h % 2 == 0) else "yuv444p"

    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-", "-i", input_video,
        "-map", "0:v:0", "-map", "1:a?",
    ]

    if encoder == "h264_nvenc":
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-gpu", str(nvenc_gpu),
            "-preset", preset,
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(cq),
        ])

    cmd.extend([
        "-pix_fmt", pix_fmt,
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-shortest",
        output_video,
    ])

    print(f"[v31] ffmpeg (using {encoder}):", " ".join(cmd[:15]) + " ...")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    import time
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v31] 使用设备: {device}")
    print(f"[v31] 视频编码器: {args.video_encoder}")

    # 加载模型
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / f"depth_anything_v2_{args.encoder}.pth"), map_location='cpu'))
    model = model.to(device).eval()
    print(f"[v31] 模型加载完成: {args.encoder}")

    # 输入视频
    video_path = Path(args.video_path)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[v31] 输入视频: {video_path.name}, {w_orig}×{h_orig}, {fps_orig:.1f} FPS, {total_frames} 帧")

    # 输出目录
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / f"{video_path.stem}_3d.mp4"

    # ---- 深度推理流尺寸 (长边518, 14倍数) ----
    input_size = 518
    scale = input_size / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = input_size
    else:
        depth_h = input_size

    # ---- DIBR 渲染流尺寸 (-1=原分辨率) ----
    dibr_h, dibr_w = h_orig, w_orig

    # 视差像素跨分辨率归一化
    max_disparity_orig = 24.0
    max_disparity_dibr = max_disparity_orig * dibr_w / w_orig

    print(f"[v31] 原始分辨率: {w_orig}×{h_orig}, FPS={fps_orig:.1f}")
    print(f"[v31] 深度模型分辨率: {depth_w}×{depth_h} (长边{input_size}, 14倍数)")
    print(f"[v31] DIBR分辨率: {dibr_w}×{dibr_h}")
    print(f"[v31] max_disparity: {max_disparity_orig} → DIBR缩放后: {max_disparity_dibr:.2f}")

    # 决定布局
    use_layout = args.layout
    if use_layout == "auto" and args.video_encoder == "h264_nvenc":
        sbs_w = w_orig * 2
        nvenc_max_w = 4096
        if sbs_w > nvenc_max_w:
            print(f"[v31] ⚠️  sbs 模式宽度 {sbs_w} 超过 NVENC 限制，自动切换到 ou 模式")
            use_layout = "ou"
        else:
            use_layout = "sbs"
    elif use_layout == "auto":
        use_layout = "sbs"

    # 计算输出尺寸
    if use_layout == "sbs":
        out_w, out_h = w_orig * 2, h_orig
        print(f"[v31] 输出模式: 并排 (SBS), 尺寸: {out_w}×{out_h}")
    else:
        out_w, out_h = w_orig, h_orig * 2
        print(f"[v31] 输出模式: 上下 (OU), 尺寸: {out_w}×{out_h}")

    # 创建 ffmpeg 写入器
    writer = create_nvenc_writer(
        ffmpeg_bin="ffmpeg",
        input_video=str(video_path),
        output_video=str(output_path),
        fps=fps_orig, out_w=out_w, out_h=out_h,
        encoder=args.video_encoder, nvenc_gpu=args.nvenc_gpu,
        preset=args.nvenc_preset, cq=args.nvenc_cq,
    )

    if writer.stdin is None:
        cap.release()
        raise RuntimeError("ffmpeg 管道初始化失败")

    # 统计
    inpaint_times = []
    iter_counts = []
    status_counts = {}
    total_times = []

    for frame_idx in tqdm(range(total_frames), ncols=80):
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
            disparity, kernel_size=15, threshold=args.sharpen_threshold,
            iterations=1, reject_margin=0.10
        )

        # DIBR 扭曲
        left_rgb = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        # 反遮挡带（GPU加速版）
        disocclusion_band = project_disocclusion_bands_gpu(
            disparity_sharp,
            min_drop=args.band_min_drop,
            min_band_width=args.band_min_width
        )
        hole_with_band = hole | disocclusion_band

        right_with_band = right_warped.clone()
        right_with_band[hole_with_band] = 0.0

        # 空洞填补 - 计时
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        img_inpainted, actual_iter, status = fast_inpaint_v30(
            right_with_band, hole_with_band.clone(), near_score,
            bg_threshold=0.3,
            edge_kernel_size=args.edge_kernel_size,
            non_edge_kernel_size=args.non_edge_kernel_size,
            max_iter=args.max_iter
        )

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inpaint_times.append((t1 - t0) * 1000)
        iter_counts.append(actual_iter)
        status_counts[status] = status_counts.get(status, 0) + 1

        # 边缘后处理
        final_right = edge_post_process(
            img_inpainted, hole_with_band,
            smooth_width=args.left_smooth_width,
            smooth_sigma=args.left_smooth_sigma,
            blend_width=args.right_blend_width
        )

        # 转换为 BGR numpy 并合成立体视频
        final_right_bgr = (final_right.cpu().numpy() * 255).astype(np.uint8)  # RGB -> BGR 顺序在下面处理
        final_right_bgr = final_right_bgr[..., ::-1].copy()  # RGB -> BGR

        if use_layout == "sbs":
            stereo_frame = np.hstack([frame_bgr, final_right_bgr])
        else:
            stereo_frame = np.vstack([frame_bgr, final_right_bgr])

        # 写入 ffmpeg（需要 RGB 顺序给 rawvideo）
        stereo_rgb = stereo_frame[..., ::-1].tobytes()
        writer.stdin.write(stereo_rgb)

        total_times.append((time.perf_counter() - frame_t0) * 1000)

    # 清理
    cap.release()
    writer.stdin.close()
    writer.wait()

    # 统计信息
    avg_time = np.mean(inpaint_times)
    avg_total_time = np.mean(total_times)
    avg_iter = np.mean(iter_counts)

    print(f"\n{'='*70}")
    print(f"✅ v31 处理完成！结果保存在: {output_path}")
    print(f"\n⏱️  性能统计:")
    print(f"  填补平均: {avg_time:.2f} ms/帧")
    print(f"  全流程平均: {avg_total_time:.2f} ms/帧 ({1000/avg_total_time:.1f} FPS)")
    print(f"\n📊 迭代次数统计:")
    print(f"  平均: {avg_iter:.1f} 次/帧")
    print(f"\n📈 填补状态统计:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count} 帧 ({count/total_frames*100:.1f}%)")
    print(f"\n💾 输出文件: {output_path.name}")
    size_mb = output_path.stat().st_size / 1024 / 1024 if output_path.exists() else 0
    print(f"  文件大小: {size_mb:.1f} MB")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
