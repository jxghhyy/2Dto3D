# python mono2stereo_with_nvds.py \
#   --video-path data/video-test/video1-Trim.mp4 \
#   --output output/output1_3d.mp4 \
#   --encoder vits \
#   --input-size 518 \
#   --max-disparity 16 \
#   --fp16 \
#   --profile-time
#
#   --enable-nvds          # 开启 NVDS 时序稳定
#   --nvds-seq-len 4       # NVDS 参考序列帧数
#   --nvds-ckpt submodules/NVDS/checkpoints/NVDS_Stabilizer.pth
#
# ============= 【集成 NVDS 版】 =============
#   ✅ 师兄优化版逻辑完全保留
#   ✅ 修复了 低分辨率DIBR尺寸不匹配 的bug
#   ✅ 新增：--enable-nvds 可选开关（默认关闭）
#   ✅ 关闭时：零性能影响，NVDS模型不加载
#   ✅ 开启时：在深度推理后插入 NVDS 时序稳定
#   ✅ 支持与 --video-model 同时开启
# ===========================================

import sys
sys.path.insert(0, '.')  # 主目录
sys.path.insert(0, './submodules/Video_Depth_Anything')  # VDA 自己的 utils 目录
sys.path.insert(0, './submodules/NVDS')  # NVDS 目录

import argparse
import glob
import os
import subprocess
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2

# 视频模型 import
try:
    from submodules.Video_Depth_Anything.video_depth_anything.video_depth_stream import VideoDepthAnything
    VIDEO_MODEL_AVAILABLE = True
except ImportError:
    VIDEO_MODEL_AVAILABLE = False

# NVDS 模型 import（懒加载，只有 --enable-nvds 时才真正 import 内部依赖）
NVDS_MODEL_AVAILABLE = False
try:
    # 先只 import 最外层，真正加载在 enable_nvds 时做
    from full_model import NVDS
    NVDS_MODEL_AVAILABLE = True
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D转3D: DepthAnything + GPU DIBR + FAST + [可选NVDS时序稳定]")
    
    # ===== 基础参数 =====
    parser.add_argument("--video-path", type=str, required=True, help="输入视频路径、视频目录或txt列表")
    parser.add_argument("--output", type=str, default="./output.mp4", help="输出视频路径（单文件模式）")
    parser.add_argument("--outdir", type=str, default="./vis_video_3d", help="输出目录（批量模式）")
    parser.add_argument("--input-size", type=int, default=518, help="DepthAnythingV2输入尺寸（长边像素，必须14倍数）")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--video-model", action="store_true", help="使用Video-Depth-Anything视频模型（内置时序一致性）")
    parser.add_argument("--metric", action="store_true", help="视频模型使用 metric 深度版本")
    parser.add_argument("--ckpt", type=str, default=None, help="模型权重路径，默认 checkpoints/video_depth_anything_{encoder}.pth")
    parser.add_argument("--fp16", action="store_true", help="CUDA下启用fp16推理")
    parser.add_argument("--warmup-iters", type=int, default=10, help="CUDA warmup轮数")
    parser.add_argument("--queue-size", type=int, default=8, help="预读队列大小（越大CPU-GPU重叠越好）")

    # ===== 深度/视差参数 =====
    parser.add_argument("--max-disparity", type=float, default=16.0, help="低分辨率下最大水平视差像素")
    parser.add_argument("--depth-mode", choices=["metric", "inverse"], default="metric", help="metric:值小更近；inverse:值大更近")
    parser.add_argument("--clip-low", type=float, default=0.01, help="深度归一化低分位")
    parser.add_argument("--clip-high", type=float, default=0.99, help="深度归一化高分位")
    parser.add_argument("--depth-smooth", type=float, default=0.0, help="时序平滑系数[0,1)")

    # ===== DIBR + 补洞 =====
    parser.add_argument("--fast-kernel", type=int, default=7, help="FAST补洞邻域核大小(奇数)")
    parser.add_argument("--fast-max-iter", type=int, default=64, help="FAST补洞最大迭代次数")
    parser.add_argument("--hole-dilate-left", type=int, default=0, help="hole向左膨胀像素，保护前景边缘（0=关闭）")

    # ===== 输出格式 =====
    parser.add_argument("--layout", choices=["sbs", "ou", "overlay", "anaglyph"], default="sbs", help="sbs并排, ou上下, overlay重合")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="overlay模式下右眼权重[0,1]")
    parser.add_argument("--anaglyph-mode", choices=["red-cyan", "color"], default="red-cyan", help="anaglyph模式：red-cyan红绿眼镜用")

    # ===== 编码 =====
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg可执行文件")
    parser.add_argument("--nvenc-preset", type=str, default="p4", help="NVENC preset")
    parser.add_argument("--nvenc-cq", type=int, default=19, help="NVENC CQ")
    parser.add_argument("--profile-time", action="store_true", help="输出各阶段平均耗时和FPS")
    parser.add_argument("--profile-total", action="store_true", help="输出脚本级端到端总耗时和总平均FPS")

    # ========== 【新增】NVDS 时序稳定参数 ==========
    parser.add_argument("--enable-nvds", action="store_true", help="开启 NVDS 时序深度稳定（默认关闭）")
    parser.add_argument("--nvds-seq-len", type=int, default=4, help="NVDS 参考序列帧数（默认4）")
    parser.add_argument("--nvds-ckpt", type=str, default="./submodules/NVDS/NVDS_checkpoints/NVDS_Stabilizer.pth", help="NVDS 模型权重路径")

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
    """统一的 CUDA 同步辅助函数"""
    if do_sync and device.type == "cuda":
        torch.cuda.synchronize()


def _normalize_depth(depth: torch.Tensor, clip_low: float, clip_high: float) -> torch.Tensor:
    # 确保是 float32，torch.quantile 只接受 float/double
    depth_float = depth.float()
    flat = depth_float.reshape(-1)
    low = torch.quantile(flat, clip_low)
    high = torch.quantile(flat, clip_high)
    denom = (high - low).clamp_min(1e-6)
    return ((depth_float - low) / denom).clamp(0.0, 1.0)


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


# ============================================================
# int64 无损 Z-buffer DIBR
# ============================================================
@torch.no_grad()
def forward_warp_right_gpu(
    left_rgb: torch.Tensor, disparity: torch.Tensor, near_score: torch.Tensor,
    stage_times: Dict[str, float], profile_sync: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w, _ = left_rgb.shape
    N = h * w  # 总像素数
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


# ============================================================
# 空洞左侧膨胀（保护前景边缘）
# ============================================================
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


def compute_aspect_preserved_size(orig_h: int, orig_w: int, long_edge: int) -> Tuple[int, int]:
    """保持宽高比，计算目标尺寸（确保两个维度都是 14 的倍数）"""
    scale = long_edge / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale / 14)) * 14
    new_w = int(round(orig_w * scale / 14)) * 14
    return new_h, new_w


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    将 tensor [H,W] 或 [C,H,W] 或 [B,C,H,W] pad 到 H,W 都是 multiple 的倍数
    返回 (padded_tensor, (orig_h, orig_w))
    """
    if tensor.dim() == 2:
        h, w = tensor.shape
        # 2D: 先加 batch 和 channel 维度变成 [1,1,H,W]，pad 完再 squeeze 回来
        tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        h, w = tensor.shape[1], tensor.shape[2]
        # 3D: 加 batch 维度变成 [1,C,H,W]
        tensor_4d = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        h, w = tensor.shape[2], tensor.shape[3]
        tensor_4d = tensor
    else:
        raise ValueError(f"不支持的维度: {tensor.dim()}")
    
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if new_h == h and new_w == w:
        return tensor, (h, w)
    
    pad_h = new_h - h
    pad_w = new_w - w
    # 右下角 pad (4D tensor 的 pad 顺序: W_left, W_right, H_top, H_bottom)
    tensor_padded = F.pad(tensor_4d, (0, pad_w, 0, pad_h), mode="reflect")
    
    # 还原维度
    if tensor.dim() == 2:
        tensor_padded = tensor_padded.squeeze(0).squeeze(0)
    elif tensor.dim() == 3:
        tensor_padded = tensor_padded.squeeze(0)
    
    return tensor_padded, (h, w)


def compose_stereo(left_u8: np.ndarray, right_u8: np.ndarray, layout: str, overlay_alpha: float) -> np.ndarray:
    if layout == "sbs":
        return np.concatenate([left_u8, right_u8], axis=1)
    if layout == "ou":
        return np.concatenate([left_u8, right_u8], axis=0)
    if layout == "anaglyph":
        h, w = left_u8.shape[:2]
        result = np.zeros_like(left_u8)
        result[..., 0] = left_u8[..., 0]
        result[..., 1] = right_u8[..., 1]
        result[..., 2] = right_u8[..., 2]
        return result
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    mixed = (1.0 - a) * left_u8.astype(np.float32) + a * right_u8.astype(np.float32)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


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


# ============================================================
# GPU 预处理（BGR→RGB + 归一化 + Resize 全GPU）
# ============================================================
@torch.no_grad()
def preprocess_gpu(
    frame_bgr: np.ndarray,
    device: torch.device,
    target_h: int,
    target_w: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    stage_times: Dict[str, float],
    profile_sync: bool,
) -> torch.Tensor:
    t0 = time.perf_counter()
    img = torch.from_numpy(frame_bgr).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_to_gpu", time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    img = img.flip(0) / 255.0  # BGR→RGB
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_bgr_to_rgb", time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    img = F.interpolate(img.unsqueeze(0), (target_h, target_w), mode="bilinear", align_corners=False)[0]
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_resize_gpu", time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)
    img = (img - mean_t) / std_t
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_normalize", time.perf_counter() - t0)
    
    return img.unsqueeze(0)


# ============================================================
# 多线程预读队列
# ============================================================
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
    model: DepthAnythingV2,
    image_gpu: torch.Tensor,
    fp16: bool,
    stage_times: Dict[str, float],
    profile_sync: bool,
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

    depth = pred[0].float()
    return depth


# ============================================================
# 【核心新增】NVDS 时序稳定器
# ============================================================
class NVDSStabilizer:
    """
    NVDS 时序深度稳定包装类
    维护一个帧缓存，每次推理用当前帧 + 历史 N-1 帧
    只保留核心推理逻辑，砍掉所有评估/迭代/混合功能
    """
    def __init__(self, model: torch.nn.Module, seq_len: int, device: torch.device):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        # 帧缓存：存储 (rgb_tensor, depth_tensor) 对，按时间顺序
        # rgb_tensor: [3, H, W]，0-1 归一化
        # depth_tensor: [H, W]，0-1 归一化
        self._cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
    
    @torch.no_grad()
    def stabilize(self, rgb_01: torch.Tensor, depth_01: torch.Tensor, 
                  stage_times: Dict[str, float], profile_sync: bool) -> torch.Tensor:
        """
        对当前帧做时序稳定
        Args:
            rgb_01: [3, H, W]，0-1 归一化 RGB
            depth_01: [H, W]，0-1 归一化深度
        Returns:
            stabilized_depth: [H, W]，时序稳定后的深度
        """
        t0 = time.perf_counter()
        device = self.device
        h, w = depth_01.shape
        
        # 1. 更新缓存（只保留最近 seq_len 帧）
        self._cache.append((rgb_01.detach().clone(), depth_01.detach().clone()))
        if len(self._cache) > self.seq_len:
            self._cache.pop(0)
        
        # 2. 构建序列：不够 seq_len 时用第一帧反向填充
        seq_rgb = []
        seq_depth = []
        # NVDS 要求：最近的帧在前（frame_0 = 当前帧，frame_1 = 前一帧，...）
        # 所以我们要逆序取缓存，然后用第一帧补前面的空缺
        cache_reversed = list(reversed(self._cache))  # 最新的在最前
        
        for i in range(self.seq_len):
            if i < len(cache_reversed):
                r, d = cache_reversed[i]
            else:
                # 用最新的帧填充（也可以用 cache_reversed[-1]，即最旧的帧）
                r, d = cache_reversed[-1]
            seq_rgb.append(r)
            seq_depth.append(d)
        
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "nvds_cache_prep", time.perf_counter() - t0)
        
        # 3. 拼接 RGBD 序列：[1, seq_len, 4, H, W]
        t0 = time.perf_counter()
        rgbd_seq = []
        for r, d in zip(seq_rgb, seq_depth):
            rgbd = torch.cat([r, d.unsqueeze(0)], dim=0)  # [4, H, W]
            rgbd_seq.append(rgbd)
        rgbd_seq = torch.stack(rgbd_seq, dim=0).unsqueeze(0).to(device)  # [1, seq_len, 4, H, W]
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "nvds_seq_concat", time.perf_counter() - t0)
        
        # 4. NVDS 推理（核心！）
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = self.model(rgbd_seq)  # [1, 1, H, W]
            pred = F.relu(pred)  # 确保深度非负
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "nvds_model_infer", time.perf_counter() - t0)
        
        # 5. 裁剪回原始尺寸（如果之前有 pad）
        stabilized_depth = pred[0, 0]  # [H, W]
        
        return stabilized_depth


def main() -> None:
    script_t0 = time.perf_counter()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("该脚本要求CUDA，享受全GPU加速。")

    torch.backends.cudnn.benchmark = True
    
    # ========== NVDS 模型加载（只有开启时才加载） ==========
    nvds_stabilizer: Optional[NVDSStabilizer] = None
    if args.enable_nvds:
        if not NVDS_MODEL_AVAILABLE:
            raise ImportError(
                "\n" + "="*70 + "\n"
                "NVDS 模型导入失败！\n"
                "请确认 submodules/NVDS 目录存在且 full_model.py 可导入\n"
                + "="*70
            )
        print(f"[mono2stereo] 📦 加载 NVDS 时序稳定模型: {args.nvds_ckpt}")
        nvds_model = NVDS()
        nvds_model = torch.nn.DataParallel(nvds_model, device_ids=[0]).cuda()
        checkpoint = torch.load(args.nvds_ckpt, map_location="cpu")
        nvds_model.load_state_dict(checkpoint)
        nvds_model.to(device).eval()
        nvds_stabilizer = NVDSStabilizer(nvds_model, seq_len=args.nvds_seq_len, device=device)
        print(f"[mono2stereo] ✅ NVDS 时序稳定已启用 (seq_len={args.nvds_seq_len})")
    
    # ========== 深度模型加载 ==========
    if args.video_model:
        if not VIDEO_MODEL_AVAILABLE:
            raise ImportError(
                "\n" + "="*70 + "\n"
                "Video-Depth-Anything 导入失败！\n"
                "请确认 submodules/Video_Depth-Anything 目录存在\n"
                "或者先不加 --video-model，用单帧模式\n"
                + "="*70
            )
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        model = VideoDepthAnything(**model_configs[args.encoder])
        checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
        ckpt = args.ckpt or f"checkpoints/{checkpoint_name}_{args.encoder}.pth"
        
        print(f"[mono2stereo] 📦 加载视频模型: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        model = model.to(device).eval()
        
        print(f"[mono2stereo] ✅ 使用 Video-Depth-Anything (官方时序一致性)")
        if args.metric:
            print(f"[mono2stereo] ✅ Metric 深度模式")
    else:
        model = DepthAnythingV2(**get_model_config(args.encoder))
        ckpt = args.ckpt or f"checkpoints/depth_anything_v2_{args.encoder}.pth"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        if args.fp16:
            model = model.half()
        print(f"[mono2stereo] ✅ 使用单帧模型 + 光流时序平滑")

    LONG_EDGE = args.input_size
    assert LONG_EDGE % 14 == 0, f"input_size必须是14的倍数，你给的是{LONG_EDGE}"

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    if args.warmup_iters > 0 and not args.video_model:
        warmup_dtype = torch.float16 if args.fp16 else torch.float32
        dummy = torch.randn(1, 3, LONG_EDGE, LONG_EDGE, device=device, dtype=warmup_dtype)
        for _ in range(args.warmup_iters):
            _ = model(dummy)
        torch.cuda.synchronize()

    files = collect_video_files(args.video_path)
    if not files:
        raise FileNotFoundError(f"未找到可处理视频: {args.video_path}")

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
        
        # 计算保持宽高比的低分辨率尺寸（14的倍数，DAv2要求）
        low_h, low_w = compute_aspect_preserved_size(h_orig, w_orig, LONG_EDGE)
        aspect_ratio = w_orig / h_orig
        
        # NVDS 要求 32 的倍数，所以需要额外 pad
        # 计算 NVDS 输入尺寸（pad 到 32 的倍数）
        nvds_h = ((low_h + 31) // 32) * 32
        nvds_w = ((low_w + 31) // 32) * 32
        need_nvds_pad = args.enable_nvds and (nvds_h != low_h or nvds_w != low_w)
        
        print(f"[mono2stereo] original: {w_orig}×{h_orig} (aspect={aspect_ratio:.2f}), FPS={fps_orig:.1f}")
        print(f"[mono2stereo] low-res processing: {low_w}×{low_h} (保持宽高比，长边={LONG_EDGE})")
        if args.enable_nvds:
            print(f"[mono2stereo] NVDS pad: → {nvds_w}×{nvds_h} (32倍数)")
        print(f"[mono2stereo] 🚀 int64无损DIBR + 前景边缘保护 (hole-dilate-left={args.hole_dilate_left})")

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

        print(f"[mono2stereo] max-disparity (low-res): {args.max_disparity:.1f} pixels "
              f"→ original: {(args.max_disparity / low_w * w_orig):.1f} pixels")

        stage_times: Dict[str, float] = {}
        processed_frames = 0
        t_video0 = time.perf_counter()
        prev_depth = None
        prev_frame_gray = None

        # 重置 NVDS 缓存（每个视频独立）
        if nvds_stabilizer is not None:
            nvds_stabilizer._cache.clear()

        reader = FrameReaderThread(cap, queue_size=args.queue_size)
        reader.start()

        try:
            while True:
                frame_t0 = time.perf_counter()

                t0 = time.perf_counter()
                ok, frame_bgr = reader.get_frame()
                _stage_add(stage_times, "read_frame_queue", time.perf_counter() - t0)
                if not ok:
                    break

                # ======================================================
                # Step 1: 深度推理（单帧模型 或 视频模型）
                # ======================================================
                t0 = time.perf_counter()
                
                # 先把原图下采样到低分辨率（✅ 修复：之前漏掉了！）
                frame_low_bgr = cv2.resize(frame_bgr, (low_w, low_h), interpolation=cv2.INTER_LINEAR)
                
                if args.video_model:
                    frame_rgb = cv2.cvtColor(frame_low_bgr, cv2.COLOR_BGR2RGB)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                        depth_low_np = model.infer_video_depth_one(
                            frame_rgb, 
                            input_size=args.input_size, 
                            device=device, 
                            fp32=(not args.fp16)
                        )
                    depth_low = torch.from_numpy(depth_low_np).to(device=device, dtype=torch.float32)
                    _stage_add(stage_times, "model_inference_video", time.perf_counter() - t0)
                else:
                    img_gpu = preprocess_gpu(
                        frame_low_bgr, device, low_h, low_w, MEAN, STD,
                        stage_times, args.profile_time
                    )
                    if args.fp16:
                        img_gpu = img_gpu.half()
                    depth_low = infer_depth_lowres(
                        model, img_gpu, args.fp16,
                        stage_times, args.profile_time
                    )
                
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "infer_depth_total", time.perf_counter() - t0)

                # ======================================================
                # Step 2: NVDS 时序稳定（可选）
                # ======================================================
                if nvds_stabilizer is not None:
                    t0 = time.perf_counter()
                    
                    # 准备 RGB: [3, H, W], 0-1 归一化
                    rgb_01 = torch.from_numpy(frame_low_bgr).permute(2, 0, 1).to(device=device, dtype=torch.float32)
                    rgb_01 = rgb_01.flip(0) / 255.0  # BGR→RGB
                    
                    # 归一化深度到 0-1（NVDS 要求输入深度在 0-1 范围）
                    depth_min, depth_max = depth_low.min(), depth_low.max()
                    if depth_max - depth_min > 1e-6:
                        depth_normalized = (depth_low - depth_min) / (depth_max - depth_min)
                    else:
                        depth_normalized = torch.zeros_like(depth_low)
                    
                    # 如果需要，pad 到 32 的倍数（NVDS 要求）
                    if need_nvds_pad:
                        rgb_padded, _ = pad_to_multiple(rgb_01, 32)
                        depth_padded, _ = pad_to_multiple(depth_normalized, 32)
                        stabilized_padded = nvds_stabilizer.stabilize(
                            rgb_padded, depth_padded, stage_times, args.profile_time
                        )
                        # 裁剪回原始低分辨率尺寸
                        depth_low = stabilized_padded[:low_h, :low_w]
                    else:
                        depth_low = nvds_stabilizer.stabilize(
                            rgb_01, depth_normalized, stage_times, args.profile_time
                        )
                    
                    # 反归一化回原始深度范围（保持和原来的深度范围一致）
                    if depth_max - depth_min > 1e-6:
                        depth_low = depth_low * (depth_max - depth_min) + depth_min
                    
                    if args.profile_time:
                        torch.cuda.synchronize()
                    _stage_add(stage_times, "nvds_stabilize_total", time.perf_counter() - t0)

                # ======================================================
                # Step 3: 时序深度平滑（光流对齐）
                # ======================================================
                if not args.video_model and args.depth_smooth > 0.0 and prev_depth is not None and prev_frame_gray is not None:
                    t0 = time.perf_counter()
                    a = float(args.depth_smooth)
                    
                    curr_frame_gray = cv2.cvtColor(frame_low_bgr, cv2.COLOR_BGR2GRAY)
                    
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame_gray, curr_frame_gray, None,
                        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                        poly_n=5, poly_sigma=1.2, flags=0
                    )
                    
                    h, w = flow.shape[:2]
                    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x = (grid_x + flow[..., 0]).astype(np.float32)
                    map_y = (grid_y + flow[..., 1]).astype(np.float32)
                    
                    prev_depth_np = prev_depth.cpu().numpy()
                    prev_depth_warped = cv2.remap(prev_depth_np, map_x, map_y, cv2.INTER_LINEAR)
                    prev_depth_warped = torch.from_numpy(prev_depth_warped).to(device=depth_low.device)
                    
                    valid_mask = (map_x > 0) & (map_x < w-1) & (map_y > 0) & (map_y < h-1)
                    valid_mask = torch.from_numpy(valid_mask).to(device=depth_low.device)
                    
                    depth_smoothed = a * depth_low + (1.0 - a) * prev_depth_warped
                    depth_low = torch.where(valid_mask, depth_smoothed, depth_low)
                    
                    _stage_add(stage_times, "depth_smooth_flow", time.perf_counter() - t0)
                    
                    prev_frame_gray = curr_frame_gray
                elif not args.video_model and args.depth_smooth > 0.0 and prev_depth is not None:
                    t0 = time.perf_counter()
                    a = float(args.depth_smooth)
                    depth_low = a * depth_low + (1.0 - a) * prev_depth
                    _stage_add(stage_times, "depth_smooth_simple", time.perf_counter() - t0)
                    prev_frame_gray = cv2.cvtColor(frame_low_bgr, cv2.COLOR_BGR2GRAY)
                elif not args.video_model and args.depth_smooth > 0.0:
                    # 第一帧初始化灰度
                    prev_frame_gray = cv2.cvtColor(frame_low_bgr, cv2.COLOR_BGR2GRAY)
                
                prev_depth = depth_low

                # ======================================================
                # Step 4: 深度转视差
                # ======================================================
                t0 = time.perf_counter()
                disparity_low, near_low = depth_to_disparity(
                    depth=depth_low,
                    max_disparity=args.max_disparity,
                    depth_mode=args.depth_mode,
                    clip_low=args.clip_low,
                    clip_high=args.clip_high,
                    stage_times=stage_times,
                    profile_sync=args.profile_time,
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "depth_to_disparity_total", time.perf_counter() - t0)

                # ======================================================
                # Step 5: GPU DIBR（✅ 现在尺寸肯定匹配了！）
                # ======================================================
                t0 = time.perf_counter()
                left_rgb_low = torch.from_numpy(frame_low_bgr).permute(2, 0, 1).to(device=device, dtype=torch.float32)
                left_rgb_low = left_rgb_low.flip(0) / 255.0  # BGR→RGB
                left_rgb_low = left_rgb_low.permute(1, 2, 0)  # CHW→HWC
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "dibr_frame_to_gpu", time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                right_rgb_low, hole_low = forward_warp_right_gpu(
                    left_rgb_low, disparity_low, near_low,
                    stage_times, args.profile_time
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "gpu_warp_total", time.perf_counter() - t0)

                # ======================================================
                # Step 6: 空洞左侧膨胀 + 补洞
                # ======================================================
                t0 = time.perf_counter()
                hole_dilated = dilate_hole_left(hole_low, args.hole_dilate_left)
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "hole_dilate_left", time.perf_counter() - t0)

                t0 = time.perf_counter()
                right_inpainted_low = fast_inpaint_gpu(
                    right_rgb_low,
                    hole_mask=hole_dilated,
                    kernel_size=args.fast_kernel,
                    max_iter=args.fast_max_iter,
                    stage_times=stage_times,
                    profile_sync=args.profile_time,
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "fast_inpaint_total", time.perf_counter() - t0)

                # ======================================================
                # Step 7: 上采样 + 合成 + 编码
                # ======================================================
                t0 = time.perf_counter()
                left_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                right_u8_low = (right_inpainted_low.clamp(0, 1) * 255.0).byte().contiguous().cpu().numpy()
                _stage_add(stage_times, "to_cpu_numpy", time.perf_counter() - t0)
                
                t0 = time.perf_counter()
                right_u8 = cv2.resize(right_u8_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                _stage_add(stage_times, "upsample_right_to_orig", time.perf_counter() - t0)

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
            if ret != 0:
                raise RuntimeError(f"ffmpeg编码失败，退出码: {ret}")

        total_elapsed = time.perf_counter() - t_video0
        avg_fps = processed_frames / max(total_elapsed, 1e-6)
        print(f"[mono2stereo] output={out_path}")
        print(f"[mono2stereo] processed_frames={processed_frames}, total_time={total_elapsed:.3f}s, avg_fps={avg_fps:.3f}")
        if args.profile_time and processed_frames > 0:
            print("[mono2stereo] 各阶段用时统计:")
            print("=" * 70)
            print(f"Depth min: {depth_low.min():.3f}, max: {depth_low.max():.3f}")
            
            top_level = [
                ("read_frame_queue", "从队列取帧 (后台预读)"),
                ("infer_depth_total", "🔹 深度推理总耗时"),
                ("nvds_stabilize_total", "🆕 NVDS时序稳定总耗时"),
                ("depth_smooth_flow", "时序深度平滑(光流对齐)"),
                ("depth_smooth_simple", "时序深度平滑(简单加权)"),
                ("depth_to_disparity_total", "🔹 深度转视差总耗时"),
                ("dibr_frame_to_gpu", "DIBR图传到GPU"),
                ("gpu_warp_total", "🔹 DIBR图像扭曲总耗时 (int64)"),
                ("hole_dilate_left", "空洞左侧膨胀(保护前景)"),
                ("fast_inpaint_total", "🔹 快速补洞总耗时"),
                ("to_cpu_numpy", "结果转回CPU"),
                ("upsample_right_to_orig", "右眼上采样回原分辨率"),
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
            
            # ---------- NVDS 内部明细 ----------
            nvds_sub = [
                ("nvds_cache_prep", "帧缓存准备 + 反向填充"),
                ("nvds_seq_concat", "RGBD序列拼接"),
                ("nvds_model_infer", "NVDS模型推理"),
            ]
            nvds_parent = "nvds_stabilize_total"
            if nvds_parent in stage_times:
                parent_time = stage_times[nvds_parent]
                print(f"\n[{nvds_parent}] 内部拆解 (父阶段总用时: {parent_time:.3f}s):")
                for key, cn_name in nvds_sub:
                    if key in stage_times:
                        t = stage_times[key]
                        ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                        print(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {cn_name}")
            
            # ---------- 其他明细 ----------
            depth_sub = [
                ("prep_to_gpu", "原图传到GPU"),
                ("prep_bgr_to_rgb", "BGR转RGB + /255 (GPU)"),
                ("prep_resize_gpu", "Resize (GPU)"),
                ("prep_normalize", "Normalize (GPU)"),
                ("model_inference", "⭐ DepthAnything模型推理(单帧)"),
                ("model_inference_video", "⭐ DepthAnything模型推理(视频多帧)"),
            ]
            depth_parent = "infer_depth_total"
            if depth_parent in stage_times:
                parent_time = stage_times[depth_parent]
                print(f"\n[{depth_parent}] 内部拆解 (父阶段总用时: {parent_time:.3f}s):")
                for key, cn_name in depth_sub:
                    if key in stage_times:
                        t = stage_times[key]
                        ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                        print(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {cn_name}")
            
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
            
            warp_sub = [
                ("warp_grid_gen", "生成网格坐标"),
                ("warp_index_prep", "准备线性索引"),
                ("warp_z_buffer", "Z-buffer冲突解决 (int64无损)"),
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
            print(f"💡 提示: 如果 'read_frame_queue' 接近 0，说明预读队列工作正常，GPU 没有等 CPU！")


if __name__ == "__main__":
    main()
