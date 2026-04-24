# 终极优化版：单目转双目 3D (极速 GPU 流水线 + int64 高保真 DIBR)
# 特性：
#   ✅ 全 GPU 预处理（彻底消除 CPU 瓶颈）
#   ✅ 动态计算宽自适应原视频比例 (严格遵守 14 倍数规则)
#   ✅ int64 无损 Z-buffer DIBR (杜绝重投影像素伪重复)
#   ✅ 前景保护：空洞左侧膨胀 + Fast Inpaint
#   ✅ 双眼同步降采样/上采样，保证视觉模糊度严格一致

import argparse
import glob
import os
import subprocess
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D转3D: 全GPU预处理 + int64 DIBR + NVENC")
    parser.add_argument("--video-path", type=str, required=True, help="输入视频路径、视频目录或txt列表")
    parser.add_argument("--output", type=str, default="./output.mp4", help="输出视频路径（单文件模式）")
    parser.add_argument("--outdir", type=str, default="./vis_video_3d", help="输出目录（批量模式）")
    parser.add_argument("--input-size", type=int, default=518, help="低分辨率基准高度（宽度自动计算，保持原比例且为14倍数）")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--ckpt", type=str, default=None, help="模型权重路径")
    parser.add_argument("--fp16", action="store_true", help="CUDA下启用fp16推理")
    parser.add_argument("--warmup-iters", type=int, default=10, help="CUDA warmup轮数")
    parser.add_argument("--queue-size", type=int, default=8, help="预读队列大小（越大CPU-GPU重叠越好）")

    parser.add_argument("--max-disparity", type=float, default=24.0,
                        help="【原分辨率等效】最大水平视差像素，内部自动缩放")
    parser.add_argument("--depth-mode", choices=["metric", "inverse"], default="metric", help="metric:值小更近；inverse:值大更近")
    parser.add_argument("--clip-low", type=float, default=0.01, help="深度归一化低分位")
    parser.add_argument("--clip-high", type=float, default=0.99, help="深度归一化高分位")
    parser.add_argument("--depth-smooth", type=float, default=0.0, help="时序平滑系数[0,1)")

    parser.add_argument("--fast-kernel", type=int, default=7, help="FAST补洞邻域核大小(奇数)")
    parser.add_argument("--fast-max-iter", type=int, default=64, help="FAST补洞最大迭代次数")
    parser.add_argument("--hole-dilate-left", type=int, default=2,
                        help="hole向左膨胀像素数，缓解inpaint啃前景边缘；0=关闭")

    parser.add_argument("--layout", choices=["sbs", "ou", "overlay"], default="sbs", help="sbs并排, ou上下, overlay重合")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="overlay模式下右眼权重[0,1]")

    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg可执行文件")
    parser.add_argument("--nvenc-preset", type=str, default="hq", help="NVENC preset (建议用hq或fast)")
    parser.add_argument("--nvenc-cq", type=int, default=19, help="NVENC CQ")
    parser.add_argument("--profile-time", action="store_true", help="输出各阶段平均耗时和FPS")
    parser.add_argument("--profile-total", action="store_true", help="输出脚本级端到端总耗时")
    return parser.parse_args()


def get_model_config(encoder: str) -> Dict:
    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    return configs[encoder]


def collect_video_files(video_path: str) -> List[str]:
    p = Path(video_path)
    if p.is_file() and p.suffix.lower() == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f.readlines() if line.strip()]
        return [x for x in files if Path(x).is_file()]
    if p.is_file():
        return [str(p)]
    files = sorted(glob.glob(os.path.join(video_path, "**", "*.mp4"), recursive=True))
    return files


def _stage_add(stage_times: Dict[str, float], key: str, dt: float) -> None:
    stage_times[key] = stage_times.get(key, 0.0) + dt


def _maybe_sync(device: torch.device, do_sync: bool) -> None:
    if do_sync and device.type == "cuda":
        torch.cuda.synchronize()


def _normalize_depth(depth: torch.Tensor, clip_low: float, clip_high: float) -> torch.Tensor:
    flat = depth.reshape(-1)
    low = torch.quantile(flat, clip_low)
    high = torch.quantile(flat, clip_high)
    denom = (high - low).clamp_min(1e-6)
    return ((depth - low) / denom).clamp(0.0, 1.0)


@torch.no_grad()
def depth_to_disparity(
    depth: torch.Tensor, max_disparity: float, depth_mode: str, clip_low: float, clip_high: float,
    stage_times: Dict[str, float], profile_sync: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = depth.device

    t0 = time.perf_counter()
    depth_norm = _normalize_depth(depth, clip_low=clip_low, clip_high=clip_high)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "depth_normalize_quantile", time.perf_counter() - t0)

    t0 = time.perf_counter()
    near = depth_norm if depth_mode == "inverse" else (1.0 - depth_norm)
    disparity = near * max_disparity
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "disparity_calc", time.perf_counter() - t0)

    return disparity, near


@torch.no_grad()
def forward_warp_right_gpu(
    left_rgb: torch.Tensor, disparity: torch.Tensor, near_score: torch.Tensor,
    stage_times: Dict[str, float], profile_sync: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w, _ = left_rgb.shape
    N = h * w
    device = left_rgb.device

    t0 = time.perf_counter()
    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    x_tgt = torch.round(xs.float() - disparity).long()
    valid = (x_tgt >= 0) & (x_tgt < w)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "warp_grid_gen", time.perf_counter() - t0)

    t0 = time.perf_counter()
    src_lin = (ys * w + xs).reshape(-1)
    tgt_lin = (ys * w + x_tgt).reshape(-1)
    valid_flat = valid.reshape(-1)
    near_flat = near_score.reshape(-1)

    src_lin = src_lin[valid_flat]
    tgt_lin = tgt_lin[valid_flat]
    near_flat = near_flat[valid_flat]
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "warp_index_prep", time.perf_counter() - t0)

    t0 = time.perf_counter()
    NEAR_BITS = 20
    src_bits = max(20, (N - 1).bit_length())
    assert NEAR_BITS + src_bits <= 62, "分辨率过高，超出 int64 编码范围"

    near_q = (near_flat.clamp(0.0, 1.0) * ((1 << NEAR_BITS) - 1)).long()
    encoded = (near_q << src_bits) | src_lin.long()

    max_encoded = torch.full((N,), -1, device=device, dtype=torch.int64)
    max_encoded.scatter_reduce_(0, tgt_lin, encoded, reduce="amax", include_self=True)
    selected = encoded == max_encoded[tgt_lin]
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "warp_z_buffer", time.perf_counter() - t0)

    t0 = time.perf_counter()
    src_sel = src_lin[selected]
    tgt_sel = tgt_lin[selected]

    left_flat = left_rgb.reshape(-1, 3)
    right_flat = torch.zeros_like(left_flat)
    right_flat[tgt_sel] = left_flat[src_sel]

    hole = torch.ones((N,), device=device, dtype=torch.bool)
    hole[tgt_sel] = False
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "warp_scatter_pixels", time.perf_counter() - t0)

    return right_flat.reshape(h, w, 3), hole.reshape(h, w)


@torch.no_grad()
def dilate_hole_left(hole: torch.Tensor, dilate_px: int) -> torch.Tensor:
    if dilate_px <= 0:
        return hole
    out = hole.clone()
    for shift in range(1, dilate_px + 1):
        out = out | torch.roll(hole, shifts=-shift, dims=1)
    out[:, -dilate_px:] = hole[:, -dilate_px:]
    return out


@torch.no_grad()
def fast_inpaint_gpu(
    image: torch.Tensor, hole_mask: torch.Tensor, kernel_size: int, max_iter: int,
    stage_times: Dict[str, float], profile_sync: bool,
) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("fast-kernel必须是奇数")

    t0 = time.perf_counter()
    img = image.clone()
    hole = hole_mask.clone()
    if not torch.any(hole):
        _stage_add(stage_times, "inpaint_skip_no_holes", time.perf_counter() - t0)
        return img
    device = img.device

    pad = kernel_size // 2
    kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=img.dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_init_kernel", time.perf_counter() - t0)

    t0 = time.perf_counter()
    iter_count = 0
    for _ in range(max_iter):
        iter_count += 1
        known = (~hole).float().unsqueeze(0).unsqueeze(0)
        if torch.count_nonzero(hole) == 0:
            break
        img_nchw = img.permute(2, 0, 1).unsqueeze(0)
        rgb_sum = F.conv2d(img_nchw * known, kernel3, padding=pad, groups=3)
        count = F.conv2d(known, kernel1, padding=pad).clamp_min(1e-6)
        avg = rgb_sum / count
        fillable = hole & (count[0, 0] > 0)
        if not torch.any(fillable):
            break
        avg_hwc = avg[0].permute(1, 2, 0)
        img[fillable] = avg_hwc[fillable]
        hole[fillable] = False
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, f"inpaint_conv_{iter_count}iter", time.perf_counter() - t0)

    t0 = time.perf_counter()
    if torch.any(hole):
        img[hole] = image[hole]
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_final_fill_remaining", time.perf_counter() - t0)

    return img


def compose_stereo(left_u8: np.ndarray, right_u8: np.ndarray, layout: str, overlay_alpha: float) -> np.ndarray:
    if layout == "sbs":
        return np.concatenate([left_u8, right_u8], axis=1)
    if layout == "ou":
        return np.concatenate([left_u8, right_u8], axis=0)
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    mixed = (1.0 - a) * left_u8.astype(np.float32) + a * right_u8.astype(np.float32)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


import sys
def create_nvenc_writer(
    ffmpeg_bin: str, input_video: str, output_video: str,
    fps: float, out_w: int, out_h: int, preset: str, cq: int,
) -> subprocess.Popen:
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-", "-i", input_video,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "h264_nvenc", "-preset", preset,
        "-rc", "vbr", "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", "-shortest", output_video,
    ]
    print("[mono2stereo] ffmpeg:", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=sys.stderr)


# ==================== 全 GPU 超快预处理合并 ====================
@torch.no_grad()
def prepare_inputs_all_gpu(
    frame_bgr: np.ndarray,
    device: torch.device,
    target_size_hw: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    stage_times: Dict[str, float],
    profile_sync: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    一步到位：将 BGR->RGB，按比例 Resize，并同时生成模型需要的 Normalize 张量
    和 DIBR 需要的 RGB 张量。所有操作 100% 在 GPU 上完成。
    """
    t0 = time.perf_counter()
    # 1. 传到 GPU (BGR)
    bgr_hwc = torch.from_numpy(frame_bgr).to(device=device, dtype=torch.float32, non_blocking=True)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_h2d", time.perf_counter() - t0)

    t0 = time.perf_counter()
    # 2. BGR -> RGB [0,1]
    rgb_hwc = bgr_hwc.flip(-1) / 255.0
    # 3. HWC -> NCHW
    rgb_nchw = rgb_hwc.permute(2, 0, 1).unsqueeze(0)
    # 4. GPU Resize 到低分辨率 (自动利用抗锯齿)
    rgb_low_nchw = F.interpolate(rgb_nchw, target_size_hw, mode="bilinear", align_corners=False)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_color_and_resize_gpu", time.perf_counter() - t0)

    t0 = time.perf_counter()
    # 5. 生成模型输入的 Normalize
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    model_input = (rgb_low_nchw - mean_t) / std_t

    # 6. 生成 DIBR 需要的低分底图
    left_rgb_low_hwc = rgb_low_nchw[0].permute(1, 2, 0).contiguous()
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_normalize", time.perf_counter() - t0)

    return model_input, left_rgb_low_hwc


class FrameReaderThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 8):
        super().__init__(daemon=True)
        self.cap = cap
        self.queue = queue.Queue(maxsize=queue_size)
        self._stop_flag = False

    def run(self):
        while not self._stop_flag and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                self.queue.put((False, None))
                break
            self.queue.put((True, frame))

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        return self.queue.get()

    def stop(self):
        self._stop_flag = True


@torch.no_grad()
def infer_depth_lowres(
    model: DepthAnythingV2, image_gpu: torch.Tensor, fp16: bool,
    stage_times: Dict[str, float], profile_sync: bool,
) -> torch.Tensor:
    device = image_gpu.device
    t0 = time.perf_counter()
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            pred = model(image_gpu)
    else:
        pred = model(image_gpu)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "model_inference", time.perf_counter() - t0)
    return pred[0].float()


def main() -> None:
    script_t0 = time.perf_counter()
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("该脚本要求CUDA。")

    torch.backends.cudnn.benchmark = True
    model = DepthAnythingV2(**get_model_config(args.encoder))
    ckpt = args.ckpt or f"checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()
    if args.fp16:
        model = model.half()

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    has_warmed_up = False

    files = collect_video_files(args.video_path)
    if not files:
        raise FileNotFoundError(f"未找到可处理视频: {args.video_path}")

    is_single_file = len(files) == 1 and args.output != "./output.mp4"
    os.makedirs(args.outdir, exist_ok=True)
    all_processed_frames = 0

    for idx, filename in enumerate(files):
        print(f"\n[mono2stereo] Progress {idx + 1}/{len(files)}: {filename}")
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print(f"[mono2stereo] 跳过，无法打开: {filename}")
            continue

        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_orig = float(cap.get(cv2.CAP_PROP_FPS))
        fps_orig = fps_orig if fps_orig > 0 else 30.0

        # --- 核心：基于高宽比动态计算宽 ---
        low_h = args.input_size
        assert low_h % 14 == 0, f"input_size({low_h}) 必须是 14 的倍数"
        aspect_ratio = w_orig / h_orig
        low_w = int(round(low_h * aspect_ratio / 14.0)) * 14
        
        max_disparity_low = args.max_disparity * low_w / w_orig

        print(f"[mono2stereo] original: {w_orig}×{h_orig}, FPS={fps_orig:.1f}")
        print(f"[mono2stereo] low-res pipeline: {low_w}×{low_h} (严格14倍数)")
        print(f"[mono2stereo] max-disparity (orig): {args.max_disparity:.1f} → (low-res): {max_disparity_low:.2f}")

        if args.warmup_iters > 0 and not has_warmed_up:
            print("[mono2stereo] CUDA Warmup 开始...")
            warmup_dtype = torch.float16 if args.fp16 else torch.float32
            dummy = torch.randn(1, 3, low_h, low_w, device=device, dtype=warmup_dtype)
            for _ in range(args.warmup_iters):
                _ = model(dummy)
            torch.cuda.synchronize()
            has_warmed_up = True

        if args.layout == "sbs":
            out_w, out_h = w_orig * 2, h_orig
        elif args.layout == "ou":
            out_w, out_h = w_orig, h_orig * 2
        else:
            out_w, out_h = w_orig, h_orig

        if is_single_file:
            out_path = args.output
        else:
            out_path = os.path.join(args.outdir, f"{Path(filename).stem}_3d.mp4")

        writer = create_nvenc_writer(
            ffmpeg_bin=args.ffmpeg_bin, input_video=filename, output_video=out_path,
            fps=fps_orig, out_w=out_w, out_h=out_h, preset=args.nvenc_preset, cq=args.nvenc_cq,
        )

        stage_times: Dict[str, float] = {}
        processed_frames = 0
        t_video0 = time.perf_counter()
        prev_depth = None

        reader = FrameReaderThread(cap, queue_size=args.queue_size)
        reader.start()

        try:
            while True:
                frame_t0 = time.perf_counter()

                t0 = time.perf_counter()
                ok, frame_bgr = reader.get_frame()
                _stage_add(stage_times, "read_frame_queue", time.perf_counter() - t0)
                if not ok: break

                # -------- Step 1 & 3: 超强合并全 GPU 预处理 --------
                t0 = time.perf_counter()
                model_input, left_rgb_low = prepare_inputs_all_gpu(
                    frame_bgr, device, (low_h, low_w), MEAN, STD,
                    stage_times, args.profile_time,
                )
                if args.fp16: model_input = model_input.half()
                _stage_add(stage_times, "preprocess_total", time.perf_counter() - t0)

                # -------- Step 2: 深度推理 --------
                t0 = time.perf_counter()
                depth_low = infer_depth_lowres(
                    model, model_input, args.fp16, stage_times, args.profile_time,
                )
                _stage_add(stage_times, "infer_depth_total", time.perf_counter() - t0)

                # -------- Step 4: 时序平滑 --------
                if args.depth_smooth > 0.0 and prev_depth is not None:
                    t0 = time.perf_counter()
                    a = float(args.depth_smooth)
                    depth_low = a * depth_low + (1.0 - a) * prev_depth
                    _maybe_sync(device, args.profile_time)
                    _stage_add(stage_times, "depth_smooth", time.perf_counter() - t0)
                prev_depth = depth_low

                # -------- Step 5: 深度转视差 --------
                t0 = time.perf_counter()
                disparity_low, near_low = depth_to_disparity(
                    depth_low, max_disparity_low, args.depth_mode, args.clip_low, args.clip_high,
                    stage_times, args.profile_time,
                )
                _stage_add(stage_times, "depth_to_disparity_total", time.perf_counter() - t0)

                # -------- Step 6: int64 Z-buffer DIBR --------
                t0 = time.perf_counter()
                right_rgb_low, hole_low = forward_warp_right_gpu(
                    left_rgb_low, disparity_low, near_low, stage_times, args.profile_time,
                )
                _stage_add(stage_times, "gpu_warp_total", time.perf_counter() - t0)

                # -------- Step 7: hole 膨胀 --------
                t0 = time.perf_counter()
                hole_dilated = dilate_hole_left(hole_low, args.hole_dilate_left)
                _maybe_sync(device, args.profile_time)
                _stage_add(stage_times, "hole_dilate", time.perf_counter() - t0)

                # -------- Step 8: Fast inpaint --------
                t0 = time.perf_counter()
                right_inpainted_low = fast_inpaint_gpu(
                    right_rgb_low, hole_dilated, args.fast_kernel, args.fast_max_iter,
                    stage_times, args.profile_time,
                )
                _stage_add(stage_times, "fast_inpaint_total", time.perf_counter() - t0)

                # -------- Step 9: 结果传回 CPU --------
                t0 = time.perf_counter()
                left_u8_low = (left_rgb_low.clamp(0, 1) * 255.0).byte().contiguous().cpu().numpy()
                right_u8_low = (right_inpainted_low.clamp(0, 1) * 255.0).byte().contiguous().cpu().numpy()
                _stage_add(stage_times, "to_cpu_numpy", time.perf_counter() - t0)

                # -------- Step 10: 均等恢复原分辨率 --------
                t0 = time.perf_counter()
                left_u8 = cv2.resize(left_u8_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                right_u8 = cv2.resize(right_u8_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                _stage_add(stage_times, "upsample_to_orig_cpu", time.perf_counter() - t0)

                # -------- Step 11 & 12: 拼接与写入 --------
                t0 = time.perf_counter()
                stereo = compose_stereo(left_u8, right_u8, args.layout, args.overlay_alpha)
                _stage_add(stage_times, "compose_stereo", time.perf_counter() - t0)

                t0 = time.perf_counter()
                writer.stdin.write(stereo.tobytes())
                _stage_add(stage_times, "write_to_ffmpeg", time.perf_counter() - t0)

                _stage_add(stage_times, "total_per_frame", time.perf_counter() - frame_t0)
                processed_frames += 1
                all_processed_frames += 1
                if processed_frames % 30 == 0:
                    print(f"[mono2stereo] {Path(filename).name} rendered {processed_frames} frames")
        finally:
            reader.stop()
            cap.release()
            writer.stdin.close()
            ret = writer.wait()
            if ret != 0: raise RuntimeError(f"ffmpeg编码失败，退出码: {ret}")

        total_elapsed = time.perf_counter() - t_video0
        avg_fps = processed_frames / max(total_elapsed, 1e-6)
        print(f"[mono2stereo] output={out_path}")
        print(f"[mono2stereo] processed_frames={processed_frames}, total_time={total_elapsed:.3f}s, avg_fps={avg_fps:.3f}")

        if args.profile_time and processed_frames > 0:
            print("\n[mono2stereo] 各阶段用时统计: (部分核心拆解)")
            print("=" * 70)
            core_stages = [
                ("read_frame_queue", "预读"),
                ("preprocess_total", "GPU 联合预处理"),
                ("infer_depth_total", "深度推理"),
                ("depth_to_disparity_total", "深度转视差"),
                ("gpu_warp_total", "int64 无损重投影"),
                ("fast_inpaint_total", "补洞修复"),
                ("upsample_to_orig_cpu", "上采样"),
                ("write_to_ffmpeg", "编码写入")
            ]
            for key, name in core_stages:
                if key in stage_times:
                    t = stage_times[key]
                    print(f"  {key:25s} {t:8.3f}s | {name}")

    if args.profile_total:
        script_elapsed = time.perf_counter() - script_t0
        script_avg_fps = all_processed_frames / max(script_elapsed, 1e-6)
        print(f"[mono2stereo][total] frames={all_processed_frames}, avg_fps={script_avg_fps:.3f}")

    # ========== 第二部分：各阶段内部细分明细 ==========
    print()
    print("🔍 【各阶段内部细分明细】")
    print("-" * 70)

    if args.profile_time and processed_frames > 0:
        print("\n[mono2stereo] 各阶段用时统计:")
        print("=" * 70)

        # ========== 顶层互斥阶段 ==========
        top_level = [
            ("read_frame_queue", "从队列取帧 (后台预读)"),
            ("preprocess_total", "⚡ 输入预处理总耗时"),
            ("infer_depth_total", "⚡ 深度推理总耗时"),
            ("depth_smooth", "时序深度平滑"),
            ("depth_to_disparity_total", "⚡ 深度转视差总耗时"),
            ("gpu_warp_total", "⚡ DIBR图像扭曲总耗时 (int64)"),
            ("hole_dilate", "hole向左膨胀"),
            ("fast_inpaint_total", "⚡ 快速补洞总耗时"),
            ("to_cpu_numpy", "结果转回CPU"),
            ("upsample_to_orig_cpu", "上采样回原分辨率(CPU)"),
            ("compose_stereo", "立体帧拼接合成"),
            ("write_to_ffmpeg", "写入ffmpeg编码"),
        ]

        top_total = sum(stage_times.get(k, 0) for k, _ in top_level)
        print(f"📊 【顶层阶段汇总】 (总和: {top_total:.3f}s)")
        for key, cn_name in top_level:
            if key in stage_times:
                t = stage_times[key]
                ratio = (t / top_total * 100.0) if top_total > 0 else 0.0
                print(f"  {key:30s} {t:8.3f}s ({ratio:5.1f}%)  |  {cn_name}")

        # ========== 内部细分明细 ==========
        print()
        print("🔍 【各阶段内部细分明细】")
        print("-" * 70)

        # 1. 预处理内部
        prep_sub = [
            ("prep_h2d", "原图BGR传到GPU"),
            ("prep_color_and_resize_gpu", "BGR→RGB + Resize (全GPU)"),
            ("prep_normalize", "Normalize标准化 (GPU)"),
        ]
        prep_parent = "preprocess_total"
        if prep_parent in stage_times:
            parent_time = stage_times[prep_parent]
            print(f"\n[{prep_parent}] 内部拆解 (父阶段总用时: {parent_time:.3f}s):")
            for key, cn_name in prep_sub:
                if key in stage_times:
                    t = stage_times[key]
                    ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                    print(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {cn_name}")

        # 2. 模型推理内部
        if "model_inference" in stage_times:
            parent_time = stage_times["infer_depth_total"]
            print(f"\n[infer_depth_total] 内部拆解 (父阶段总用时: {parent_time:.3f}s):")
            t = stage_times["model_inference"]
            ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
            print(f"  ├─ {'model_inference':25s} {t:8.3f}s ({ratio:5.1f}%)  |  ⭐ DepthAnything模型推理")

        # 3. 深度转视差内部
        disp_sub = [
            ("depth_normalize_quantile", "分位数裁剪归一化"),
            ("disparity_calc", "计算视差值"),
        ]
        disp_parent = "depth_to_disparity_total"
        if disp_parent in stage_times:
            parent_time = stage_times[disp_parent]
            print(f"\n[{disp_parent}] 内部拆解 (父阶段总用时: {parent_time:.3f}s):")
            for key, cn_name in disp_sub:
                if key in stage_times:
                    t = stage_times[key]
                    ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                    print(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {cn_name}")

        # 4. int64 DIBR扭曲内部
        warp_sub = [
            ("warp_grid_gen", "生成网格坐标"),
            ("warp_index_prep", "准备线性索引"),
            ("warp_z_buffer", "无损 Z-buffer (int64编码)"),
            ("warp_scatter_pixels", "像素散射"),
        ]
        warp_parent = "gpu_warp_total"
        if warp_parent in stage_times:
            parent_time = stage_times[warp_parent]
            print(f"\n[{warp_parent}] 内部拆解 (父阶段总用时: {parent_time:.3f}s):")
            for key, cn_name in warp_sub:
                if key in stage_times:
                    t = stage_times[key]
                    ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                    print(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {cn_name}")

        # 5. 补洞内部
        inpaint_parent = "fast_inpaint_total"
        if inpaint_parent in stage_times:
            parent_time = stage_times[inpaint_parent]
            print(f"\n[{inpaint_parent}] 内部拆解 (父阶段总用时: {parent_time:.3f}s):")
            inpaint_keys = [k for k in stage_times if k.startswith("inpaint_")]
            for key in sorted(inpaint_keys):
                t = stage_times[key]
                ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                if key == "inpaint_skip_no_holes":
                    cn_name = "跳过(无空洞)"
                elif key == "inpaint_init_kernel":
                    cn_name = "初始化卷积核"
                elif key == "inpaint_final_fill_remaining":
                    cn_name = "剩余空洞兜底"
                elif key.startswith("inpaint_conv_"):
                    iters = key.replace("inpaint_conv_", "").replace("iter", "")
                    cn_name = f"卷积迭代 ({iters}次)"
                else:
                    cn_name = ""
                print(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {cn_name}")

        print()
        print("=" * 70)
        print(f"📈 处理帧数: {processed_frames}, 平均FPS: {processed_frames / total_elapsed:.2f}")
        print(f"💡 提示: 如果 'read_frame_queue' 耗时极短，说明预读队列跑满了，显卡吃满了！")

if __name__ == "__main__":
    main()