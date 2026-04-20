# python mono2stereo_lower_fastinpaint.py \
#   --video-path data/video-test/video1-Trim.mp4 \
#   --output output/output1_3d.mp4 \
#   --encoder vits \
#   --input-size 518 \
#   --max-disparity 16 \
#   --fp16 \
#   --profile-time

import argparse
import glob
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
from submodules.depth.dav2.depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, MyResize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D转3D: 低分辨率DepthAnything + GPU DIBR + GPU FastInpaint")
    parser.add_argument("--video-path", type=str, required=True, help="输入视频路径、视频目录或txt列表")
    parser.add_argument("--output", type=str, default="./output.mp4", help="输出视频路径（单文件模式）")
    parser.add_argument("--outdir", type=str, default="./vis_video_3d", help="输出目录（批量模式）")
    parser.add_argument("--input-size", type=int, default=518, help="DepthAnythingV2输入尺寸（正方形，必须14倍数）")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--ckpt", type=str, default=None, help="模型权重路径，默认 checkpoints/depth_anything_v2_{encoder}.pth")
    parser.add_argument("--fp16", action="store_true", help="CUDA下启用fp16推理")
    parser.add_argument("--warmup-iters", type=int, default=10, help="CUDA warmup轮数")

    parser.add_argument("--max-disparity", type=float, default=16.0, help="低分辨率下最大水平视差像素")
    parser.add_argument("--depth-mode", choices=["metric", "inverse"], default="metric", help="metric:值小更近；inverse:值大更近")
    parser.add_argument("--clip-low", type=float, default=0.01, help="深度归一化低分位")
    parser.add_argument("--clip-high", type=float, default=0.99, help="深度归一化高分位")
    parser.add_argument("--depth-smooth", type=float, default=0.0, help="时序平滑系数[0,1)")

    parser.add_argument("--fast-kernel", type=int, default=7, help="FAST补洞邻域核大小(奇数)")
    parser.add_argument("--fast-max-iter", type=int, default=64, help="FAST补洞最大迭代次数")

    parser.add_argument("--layout", choices=["sbs", "ou", "overlay"], default="sbs", help="sbs并排, ou上下, overlay重合")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="overlay模式下右眼权重[0,1]")

    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg可执行文件")
    parser.add_argument("--nvenc-preset", type=str, default="p4", help="NVENC preset")
    parser.add_argument("--nvenc-cq", type=int, default=19, help="NVENC CQ")
    parser.add_argument("--profile-time", action="store_true", help="输出各阶段平均耗时和FPS")
    parser.add_argument("--profile-total", action="store_true", help="输出脚本级端到端总耗时和总平均FPS")
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


def _normalize_depth(depth: torch.Tensor, clip_low: float, clip_high: float) -> torch.Tensor:
    flat = depth.reshape(-1)
    low = torch.quantile(flat, clip_low)
    high = torch.quantile(flat, clip_high)
    denom = (high - low).clamp_min(1e-6)
    return ((depth - low) / denom).clamp(0.0, 1.0)


@torch.no_grad()
def depth_to_disparity(
    depth: torch.Tensor, max_disparity: float, depth_mode: str, clip_low: float, clip_high: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth_norm = _normalize_depth(depth, clip_low=clip_low, clip_high=clip_high)
    near = depth_norm if depth_mode == "inverse" else (1.0 - depth_norm)
    disparity = near * max_disparity
    return disparity, near


@torch.no_grad()
def forward_warp_right_gpu(
    left_rgb: torch.Tensor, disparity: torch.Tensor, near_score: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w, _ = left_rgb.shape
    device = left_rgb.device

    ys = torch.arange(h, device=device).view(h, 1).expand(h, w)
    xs = torch.arange(w, device=device).view(1, w).expand(h, w)
    x_tgt = torch.round(xs.float() - disparity).long()
    valid = (x_tgt >= 0) & (x_tgt < w)

    src_lin = (ys * w + xs).reshape(-1)
    tgt_lin = (ys * w + x_tgt).reshape(-1)
    valid_flat = valid.reshape(-1)
    near_flat = near_score.reshape(-1)

    src_lin = src_lin[valid_flat]
    tgt_lin = tgt_lin[valid_flat]
    near_flat = near_flat[valid_flat]

    tie_break = src_lin.float() / float(h * w) * 1e-3
    score = near_flat + tie_break

    max_score = torch.full((h * w,), -1e9, device=device, dtype=torch.float32)
    max_score.scatter_reduce_(0, tgt_lin, score, reduce="amax", include_self=True)
    selected = (score - max_score[tgt_lin]).abs() < 1e-8

    src_sel = src_lin[selected]
    tgt_sel = tgt_lin[selected]
    left_flat = left_rgb.reshape(-1, 3)
    right_flat = torch.zeros_like(left_flat)
    right_flat[tgt_sel] = left_flat[src_sel]

    hole = torch.ones((h * w,), device=device, dtype=torch.bool)
    hole[tgt_sel] = False
    return right_flat.reshape(h, w, 3), hole.reshape(h, w)


@torch.no_grad()
def fast_inpaint_gpu(image: torch.Tensor, hole_mask: torch.Tensor, kernel_size: int, max_iter: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("fast-kernel必须是奇数")

    img = image.clone()
    hole = hole_mask.clone()
    if not torch.any(hole):
        return img

    pad = kernel_size // 2
    kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=img.device, dtype=img.dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)

    for _ in range(max_iter):
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

    if torch.any(hole):
        img[hole] = image[hole]
    return img


def compose_stereo(left_u8: np.ndarray, right_u8: np.ndarray, layout: str, overlay_alpha: float) -> np.ndarray:
    if layout == "sbs":
        return np.concatenate([left_u8, right_u8], axis=1)
    if layout == "ou":
        return np.concatenate([left_u8, right_u8], axis=0)
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    mixed = (1.0 - a) * left_u8.astype(np.float32) + a * right_u8.astype(np.float32)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def _stage_add(stage_times: Dict[str, float], key: str, dt: float) -> None:
    stage_times[key] = stage_times.get(key, 0.0) + dt


def create_nvenc_writer(
    ffmpeg_bin: str,
    input_video: str,
    output_video: str,
    fps: float,
    out_w: int,
    out_h: int,
    preset: str,
    cq: int,
) -> subprocess.Popen:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{out_w}x{out_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        input_video,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "h264_nvenc",
        "-preset",
        preset,
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        output_video,
    ]
    print("[mono2stereo] ffmpeg:", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


@torch.no_grad()
def infer_depth_lowres(
    model: DepthAnythingV2,
    transform: Compose,
    frame_bgr: np.ndarray,
    device: torch.device,
    fp16: bool,
    input_size: int,
) -> torch.Tensor:
    """
    低分辨率深度推断：先把原图resize到正方形input_size×input_size，推理后输出低分辨率深度
    最后再在上采样阶段还原到原尺寸
    """
    h_orig, w_orig = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": rgb})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    if fp16 and device.type == "cuda":
        image = image.half()

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            pred = model(image)
    else:
        pred = model(image)

    # 输出就是input_size×input_size低分辨率深度，不在这里上采样
    depth = pred[0].float()
    return depth


def main() -> None:
    script_t0 = time.perf_counter()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("该脚本要求CUDA，享受全GPU加速。")

    torch.backends.cudnn.benchmark = True
    model = DepthAnythingV2(**get_model_config(args.encoder))
    ckpt = args.ckpt or f"checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()
    if args.fp16:
        model = model.half()

    # 低分辨率正方形输入变换，和你本地版本一致，保证是14的倍数
    LOW_RES = args.input_size
    assert LOW_RES % 14 == 0, f"input_size必须是14的倍数，你给的是{LOW_RES}"

    transform = Compose([
        MyResize(
            width=LOW_RES, height=LOW_RES,
            resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14,
            resize_method="minimal",
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if args.warmup_iters > 0:
        warmup_dtype = torch.float16 if args.fp16 else torch.float32
        dummy = torch.randn(1, 3, LOW_RES, LOW_RES, device=device, dtype=warmup_dtype)
        for _ in range(args.warmup_iters):
            _ = model(dummy)
        torch.cuda.synchronize()

    files = collect_video_files(args.video_path)
    if not files:
        raise FileNotFoundError(f"未找到可处理视频: {args.video_path}")

    # 判断是单文件模式还是批量模式
    is_single_file = len(files) == 1 and args.output != "./output.mp4"
    os.makedirs(args.outdir, exist_ok=True)
    all_processed_frames = 0

    for idx, filename in enumerate(files):
        print(f"[mono2stereo] Progress {idx + 1}/{len(files)}: {filename}")
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print(f"[mono2stereo] 跳过，无法打开: {filename}")
            continue

        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_orig = float(cap.get(cv2.CAP_PROP_FPS))
        fps_orig = fps_orig if fps_orig > 0 else 30.0
        print(f"[mono2stereo] original: {w_orig}×{h_orig}, FPS={fps_orig:.1f}")
        print(f"[mono2stereo] low-res processing: {LOW_RES}×{LOW_RES}")

        # 计算输出尺寸
        if args.layout == "sbs":
            out_w, out_h = w_orig * 2, h_orig
        elif args.layout == "ou":
            out_w, out_h = w_orig, h_orig * 2
        else:
            out_w, out_h = w_orig, h_orig

        # 确定输出路径
        if is_single_file:
            out_path = args.output
        else:
            out_path = os.path.join(args.outdir, f"{Path(filename).stem}_3d.mp4")

        writer = create_nvenc_writer(
            ffmpeg_bin=args.ffmpeg_bin,
            input_video=filename,
            output_video=out_path,
            fps=fps_orig,
            out_w=out_w,
            out_h=out_h,
            preset=args.nvenc_preset,
            cq=args.nvenc_cq,
        )
        if writer.stdin is None:
            cap.release()
            raise RuntimeError("ffmpeg管道初始化失败")

        # 3D效果强度：低分辨率下max_disparity对应最终原分辨率的 (max_disparity / LOW_RES * w_orig)
        # 所以只需要按比例设定低分辨率下的max_disparity就能保持一致的视觉强度
        print(f"[mono2stereo] max-disparity (low-res): {args.max_disparity:.1f} pixels "
              f"→ original: {(args.max_disparity / LOW_RES * w_orig):.1f} pixels")

        stage_times: Dict[str, float] = {}
        processed_frames = 0
        t_video0 = time.perf_counter()
        prev_depth = None

        try:
            while cap.isOpened():
                frame_t0 = time.perf_counter()

                t0 = time.perf_counter()
                ok, frame_bgr = cap.read()
                _stage_add(stage_times, "read_frame", time.perf_counter() - t0)
                if not ok:
                    break

                # Step 1: 低分辨率深度推断 (速度快！)
                t0 = time.perf_counter()
                depth_low = infer_depth_lowres(
                    model, transform, frame_bgr, device, args.fp16, args.input_size
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "infer_depth_low", time.perf_counter() - t0)

                # Step 2: 时序深度平滑（可选）
                if args.depth_smooth > 0.0 and prev_depth is not None:
                    t0 = time.perf_counter()
                    a = float(args.depth_smooth)
                    depth_low = a * depth_low + (1.0 - a) * prev_depth
                    _stage_add(stage_times, "depth_smooth", time.perf_counter() - t0)
                prev_depth = depth_low

                # Step 3: 深度转视差（依然低分辨率）
                t0 = time.perf_counter()
                disparity_low, near_low = depth_to_disparity(
                    depth=depth_low,
                    max_disparity=args.max_disparity,
                    depth_mode=args.depth_mode,
                    clip_low=args.clip_low,
                    clip_high=args.clip_high,
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "depth_to_disparity", time.perf_counter() - t0)

                # Step 4: 左图resize到低分辨率，GPU DIBR（小尺寸计算更快）
                t0 = time.perf_counter()
                frame_low_bgr = cv2.resize(frame_bgr, (LOW_RES, LOW_RES), interpolation=cv2.INTER_LINEAR)
                left_rgb_low = torch.from_numpy(
                    cv2.cvtColor(frame_low_bgr, cv2.COLOR_BGR2RGB)
                ).to(device=device, dtype=torch.float32) / 255.0
                right_rgb_low, hole_low = forward_warp_right_gpu(left_rgb_low, disparity_low, near_low)
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "gpu_warp_low", time.perf_counter() - t0)

                # Step 5: GPU低分辨率快速补洞（小尺寸更快）
                t0 = time.perf_counter()
                right_inpainted_low = fast_inpaint_gpu(
                    right_rgb_low,
                    hole_mask=hole_low,
                    kernel_size=args.fast_kernel,
                    max_iter=args.fast_max_iter,
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "fast_inpaint_low", time.perf_counter() - t0)

                # Step 6: 左右眼同时上采样到原分辨率（CPU OpenCV resize，很快）
                t0 = time.perf_counter()
                left_u8_low = (left_rgb_low.clamp(0, 1) * 255.0).byte().cpu().numpy()
                right_u8_low = (right_inpainted_low.clamp(0, 1) * 255.0).byte().cpu().numpy()

                left_u8 = cv2.resize(left_u8_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                right_u8 = cv2.resize(right_u8_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

                # 拼接输出
                stereo = compose_stereo(left_u8, right_u8, args.layout, args.overlay_alpha)
                _stage_add(stage_times, "upsample_compose", time.perf_counter() - t0)

                # Step 7: 写入ffmpeg编码器
                t0 = time.perf_counter()
                writer.stdin.write(stereo.tobytes())
                _stage_add(stage_times, "write_encoder", time.perf_counter() - t0)

                _stage_add(stage_times, "total_per_frame", time.perf_counter() - frame_t0)
                processed_frames += 1
                all_processed_frames += 1
                if processed_frames % 30 == 0:
                    print(f"[mono2stereo] {Path(filename).name} rendered {processed_frames} frames")
        finally:
            cap.release()
            writer.stdin.close()
            ret = writer.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg编码失败，退出码: {ret}")

        total_elapsed = time.perf_counter() - t_video0
        avg_fps = processed_frames / max(total_elapsed, 1e-6)
        print(f"[mono2stereo] output={out_path}")
        print(f"[mono2stereo] processed_frames={processed_frames}, total_time={total_elapsed:.3f}s, avg_fps={avg_fps:.3f}")
        if args.profile_time and processed_frames > 0:
            print("[mono2stereo] average stage latency (ms/frame):")
            for key in [
                "read_frame",
                "infer_depth_low",
                "depth_smooth",
                "depth_to_disparity",
                "gpu_warp_low",
                "fast_inpaint_low",
                "upsample_compose",
                "write_encoder",
                "total_per_frame",
            ]:
                if key in stage_times:
                    print(f"  - {key}: {(stage_times[key] / processed_frames) * 1000.0:.3f} ms")

    if args.profile_total:
        script_elapsed = time.perf_counter() - script_t0
        script_avg_fps = all_processed_frames / max(script_elapsed, 1e-6)
        print(
            f"[mono2stereo][total] videos={len(files)}, frames={all_processed_frames}, "
            f"total_time={script_elapsed:.3f}s, avg_fps={script_avg_fps:.3f}"
        )


if __name__ == "__main__":
    main()
