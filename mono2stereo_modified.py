# python mono2stereo_lower_fastinpaint_time_gpu_video_zc.py \
#   --video-path data/video-test/video1-Trim.mp4 \
#   --output output/output1_3d.mp4 \
#   --encoder vits \
#   --input-size 518 \
#   --max-disparity 16 \
#   --fp16 \
#   --profile-time
#
# ==================== 【师兄优化版】 ====================
#   ✅ 原版逻辑完全保留，只修改了 DIBR + inpaint 两块
#   ✅ DIBR: int64 无损 Z-buffer（彻底消除像素冲突伪影）
#   ✅ 新增：空洞左侧膨胀（保护前景边缘不被 inpaint 侵蚀）
#
# ==================== 【终极优化：PLAN A-E 时序稳定】 ====================
# 五重时序稳定（不损失速度，全部在 "深度推理分辨率" 上完成）：
#
#   ✅ Plan A —— 分位数 EMA：消除 DepthAnything V2 每帧 scale/shift
#                漂移导致的整帧 flicker（几乎零开销）
#
#   ✅ Plan B —— RGB 引导的逐像素自适应 EMA：
#                静止区域强平滑，运动区域不平滑（无 ghosting）
#
#   ✅ Plan C —— 平滑作用在 "归一化后的 near 空间"，
#                与渲染量直接相关，避免归一化区间逐帧抖动抵消平滑
#
#   ✅ Plan D —— 低分辨率光流补偿的 Motion-aligned EMA：
#                CPU Farneback 在小分辨率算后向光流，
#                grid_sample 反向 warp prev_near 到当前帧空间位置后再 EMA。
#                与 B 协同：D 修几何对齐、B 修 flow 失败兜底。
#
#   ✅ Plan E —— 滑窗逐像素中值（K=3）：
#                抗单帧极值噪声；与 EMA 正交，叠加使用效果最稳。
#
# 额外新增：深度推理 / DIBR 渲染分辨率彻底解耦（--dibr-size 参数）
# =============================================================================

# ==================== 【Modified: 深度感知空洞修补】 ====================
#   ✅ fast_inpaint_gpu 现在接受 near_score 参数
#   ✅ 填充时空洞优先使用背景像素（远的像素）
#   ✅ 通过深度加权平均，避免前景颜色污染背景区域
# =============================================================================

import sys
sys.path.insert(0, '.')  # 主目录
sys.path.insert(0, './submodules/Video_Depth_Anything')  # VDA 自己的 utils 目录

import argparse
import functools
import glob
import os
import subprocess
import time
import threading
import queue
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2

# 视频模型 import（放在这里，有 import 错误也不影响单帧模式）
try:
    from submodules.Video_Depth_Anything.video_depth_anything.video_depth_stream import VideoDepthAnything
    VIDEO_MODEL_AVAILABLE = True
except ImportError:
    VIDEO_MODEL_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D转3D: 低分辨率DepthAnything + GPU DIBR + GPU FastInpaint + 时序稳定A-E")
    parser.add_argument("--video-path", type=str, required=True, help="输入视频路径、视频目录或txt列表")
    parser.add_argument("--output", type=str, default="./output.mp4", help="输出视频路径（单文件模式）")
    parser.add_argument("--outdir", type=str, default="./vis_video_3d", help="输出目录（批量模式）")

    # ---- 分辨率 ----
    parser.add_argument("--input-size", type=int, default=518, help="深度模型输入基准高度（必须为14的倍数）")
    parser.add_argument("--dibr-size", type=int, default=0, help="DIBR渲染基准高度（-1=原分辨率，0=和input-size一致）")

    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--video-model", action="store_true", help="使用Video-Depth-Anything视频模型（内置时序一致性）")
    parser.add_argument("--metric", action="store_true", help="视频模型使用 metric 深度版本")
    parser.add_argument("--ckpt", type=str, default=None, help="模型权重路径，默认 checkpoints/video_depth_anything_{encoder}.pth")
    parser.add_argument("--fp16", action="store_true", help="CUDA下启用fp16推理")
    parser.add_argument("--warmup-iters", type=int, default=10, help="CUDA warmup轮数")
    parser.add_argument("--queue-size", type=int, default=8, help="预读队列大小（越大CPU-GPU并行越好）")

    parser.add_argument("--max-disparity", type=float, default=16.0, help="原分辨率等效最大水平视差像素，内部自动按DIBR尺寸缩放")
    parser.add_argument("--depth-mode", choices=["metric", "inverse"], default="inverse", help="metric:值小更近；inverse:值大更近")
    parser.add_argument("--clip-low", type=float, default=0.01, help="深度归一化低分位")
    parser.add_argument("--clip-high", type=float, default=0.99, help="深度归一化高分位")

    # ---- 时序稳定化（Plan A / B / C / D / E） ----
    parser.add_argument("--depth-smooth", type=float, default=0.0,
                        help="【Plan B+C】near空间EMA强度（前帧权重；0=禁用，0.6推荐）")
    parser.add_argument("--quantile-smooth", type=float, default=0.0,
                        help="【Plan A】分位数EMA强度（前帧权重；0=禁用，0.8推荐，几乎零开销）")
    parser.add_argument("--rgb-motion-sigma", type=float, default=0.0,
                        help="【Plan B】RGB帧差sigma；越小越敏感于运动；0=禁用自适应（退化为均匀EMA）")

    # Plan D: 光流对齐
    parser.add_argument("--flow-align", action="store_true",
                        help="【Plan D】启用CPU Farneback光流补偿的motion-aligned EMA（约+1~3ms/帧）")
    parser.add_argument("--flow-height", type=int, default=144,
                        help="【Plan D】光流计算分辨率的基准高度；越大越精确越慢，144推荐）")

    # Plan E: 滑窗中值
    parser.add_argument("--median-window", type=int, default=1,
                        help="【Plan E】滑窗中值滤波窗口大小（1=禁用；3推荐；最大7）")

    parser.add_argument("--fast-kernel", type=int, default=11, help="FAST补洞邻域核大小(奇数)")
    parser.add_argument("--fast-max-iter", type=int, default=64, help="FAST补洞最大迭代次数")
    parser.add_argument("--hole-dilate-left", type=int, default=0, help="hole向左膨胀像素，保护前景边缘（0=关闭）")

    # ---- 深度感知补洞参数 ----
    parser.add_argument("--bg-beta", type=float, default=6.0,
                        help="背景偏好系数：越大越偏好背景像素填充")

    parser.add_argument("--layout", choices=["sbs", "ou", "overlay", "anaglyph"], default="sbs", help="sbs并排, ou上下, overlay重合")
    parser.add_argument("--overlay-alpha", type=float, default=0.5, help="overlay模式下右眼权重[0,1]")
    parser.add_argument("--anaglyph-mode", choices=["red-cyan", "color"], default="red-cyan", help="anaglyph模式：red-cyan红绿眼镜用")

    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg可执行文件")
    parser.add_argument("--video-encoder", type=str, default="h264_nvenc",
        choices=["h264_nvenc", "libx264", "h264_vaapi"],
        help="视频编码器: h264_nvenc=NVIDIA硬件编码, libx264=CPU编码")
    parser.add_argument("--nvenc-gpu", type=int, default=0, help="h264_nvenc 使用的 GPU ID (避免和 PyTorch 冲突)")
    parser.add_argument("--nvenc-preset", type=str, default="medium", help="NVENC preset (default, slow, medium, fast, hp, hq, bd, ll, llhq, llhp, lossless, losslesshp)")
    parser.add_argument("--nvenc-cq", type=int, default=19, help="NVENC CQ (h264_nvenc) / CRF (libx264)")
    parser.add_argument("--profile-time", action="store_true", help="输出各阶段平均耗时和FPS")
    parser.add_argument("--profile-total", action="store_true", help="输出脚本级端到端总耗时和总平均FPS")
    parser.add_argument("--profile-output", type=str, default=None, help="性能统计保存到文件路径")
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


@functools.lru_cache(maxsize=8)
def _get_inpaint_kernels(kernel_size: int, device: str, dtype: str):
    """缓存 inpaint kernel，避免每帧重建"""
    device = torch.device(device)
    dtype = getattr(torch, dtype)
    pad = kernel_size // 2
    kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)
    return kernel1, kernel3


# =============================================================================
# Plan D: 光流估计 + 反向 Warp
# =============================================================================

class OpticalFlowEstimator:
    """
    CPU Farneback 后向光流估计器（在小分辨率上算，速度 ~1~3ms/帧）。

    "后向光流" 定义：对当前帧的每个像素 (x, y)，flow[(y, x)] = (dx, dy)，
                    使得 prev(x + dx, y + dy) ≈ curr(x, y)。
    用此光流配合 grid_sample 即可把 prev_* 反向 warp 到当前帧空间位置。
    """

    def __init__(self, flow_height: int = 144) -> None:
        self.flow_height = int(flow_height)
        self._prev_gray_cpu: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray_cpu = None

    @torch.no_grad()
    def estimate_backward_flow(
            self,
            rgb_curr: torch.Tensor,
            target_size_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """
        rgb_curr: [3, H, W] on GPU, [0,1]
        target_size_hw: (H_t, W_t) - 返回光流的目标尺寸（通常 = 深度推理分辨率）
        return: [2, H_t, W_t] (dx, dy) 像素位移，或 None（首帧）
        """
        device = rgb_curr.device
        H_in, W_in = rgb_curr.shape[-2:]
        flow_h = self.flow_height
        flow_w = max(2, int(round(W_in * flow_h / H_in)))

        # 灰度化 + 下采样
        gray = (0.299 * rgb_curr[0] + 0.587 * rgb_curr[1] + 0.114 * rgb_curr[2])
        gray_small = F.interpolate(
            gray[None, None], size=(flow_h, flow_w),
            mode="bilinear", align_corners=False,
        )[0, 0]
        gray_cpu = (gray_small.clamp(0, 1) * 255.0).byte().cpu().numpy()

        if self._prev_gray_cpu is None:
            self._prev_gray_cpu = gray_cpu
            return None

        # 后向光流：从 curr 到 prev 的位移
        flow_small = cv2.calcOpticalFlowFarneback(
            gray_cpu, self._prev_gray_cpu, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )  # [flow_h, flow_w, 2] float32
        self._prev_gray_cpu = gray_cpu

        flow_t = torch.from_numpy(flow_small).to(device).permute(2, 0, 1)  # [2, h, w]
        H_t, W_t = target_size_hw
        flow_up = F.interpolate(
            flow_t[None], size=(H_t, W_t), mode="bilinear", align_corners=False,
        )[0]
        # 上采样改变了像素尺度，光流值要相应缩放
        flow_up[0] *= W_t / float(flow_w)
        flow_up[1] *= H_t / float(flow_h)
        return flow_up


@torch.no_grad()
def backward_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    image: [H, W] 或 [C, H, W]
    flow:  [2, H, W] 像素单位的后向光流（curr → prev 偏移）
    return:   与 image 同 shape 的 warped 结果（border 填充）
    """
    H, W = flow.shape[-2:]
    device = flow.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=flow.dtype),
        torch.arange(W, device=device, dtype=flow.dtype),
        indexing="ij",
    )
    x_sample = xx + flow[0]
    y_sample = yy + flow[1]
    # grid_sample 用归一化坐标 [-1, 1], align_corners=False
    x_norm = 2.0 * (x_sample + 0.5) / W - 1.0
    y_norm = 2.0 * (y_sample + 0.5) / H - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)[None]  # [1, H, W, 2]

    if image.dim() == 2:
        warped = F.grid_sample(
            image[None, None], grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        )[0, 0]
    else:
        warped = F.grid_sample(
            image[None], grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        )[0]
    return warped


# =============================================================================
# 时序稳定器模块（Plan A + B + C + D + E 一体）
# =============================================================================

class TemporalDepthStabilizer:
    """
    在 "深度模型输出分辨率" 的 'near'（1=最近）空间内统一处理：

      Plan A —— 分位数 EMA（消除整帧尺度漂移）
      Plan B —— RGB 引导的逐像素自适应 EMA
      Plan C —— 平滑作用在归一化后空间
      Plan D —— 光流补偿的 Motion-aligned EMA
                 （warp prev_near & prev_rgb 后再做 B+C）
      Plan E —— 滑窗逐像素中值（post-EMA）

    每帧输出：稳定后的 near。
    用法：
        stab = TemporalDepthStabilizer(...)
        stab.reset()
        near_smooth = stab.step(depth_raw, rgb_depth, ...)
    """

    def __init__(
            self,
            depth_mode: str,
            clip_low: float,
            clip_high: float,
            depth_smooth: float,
            quantile_smooth: float,
            rgb_motion_sigma: float,
            flow_align_enabled: bool,
            flow_height: int,
            median_window: int,
    ) -> None:
        self.depth_mode = depth_mode
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.depth_smooth = float(depth_smooth)
        self.quantile_smooth = float(quantile_smooth)
        self.rgb_motion_sigma = float(rgb_motion_sigma)

        # Plan D
        self.flow_align_enabled = bool(flow_align_enabled)
        self.flow_estimator: Optional[OpticalFlowEstimator] = (
            OpticalFlowEstimator(flow_height) if self.flow_align_enabled else None
        )

        # Plan E
        self.median_window = max(1, int(median_window))
        self._near_history: Deque[torch.Tensor] = deque(maxlen=self.median_window)

        # Plan A 状态（标量 EMA）
        self._low_ema: Optional[torch.Tensor] = None
        self._high_ema: Optional[torch.Tensor] = None

        # Plan B / D 状态（前帧低分图）
        self._prev_near: Optional[torch.Tensor] = None
        self._prev_rgb: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """新视频前调用，避免上一段视频状态污染。"""
        self._low_ema = None
        self._high_ema = None
        self._prev_near = None
        self._prev_rgb = None
        if self.flow_estimator is not None:
            self.flow_estimator.reset()
        self._near_history.clear()

    @torch.no_grad()
    def step(
            self,
            depth_raw: torch.Tensor,
            rgb_depth: torch.Tensor,
            stage_times: Dict[str, float],
            profile_sync: bool,
    ) -> torch.Tensor:
        """
        depth_raw: [H, W]    深度模型原始输出
        rgb_depth: [3, H, W] 同分辨率 RGB [0,1]
        return:    [H, W]    稳定后的 near
        """
        device = depth_raw.device

        # ---------- Plan A: 分位数 EMA + 归一化 ----------
        t0 = time.perf_counter()
        # M-1: 随机采样 16k 像素算 quantile，误差可忽略，速度提升 ~100x
        flat = depth_raw.reshape(-1)
        n_total = flat.numel()
        sample_size = 16384
        idx = torch.randint(0, n_total, (sample_size,), device=flat.device)
        sample = flat[idx]
        q_vals = torch.quantile(sample, torch.tensor(
            [self.clip_low, self.clip_high], device=flat.device,
        ))
        low_curr, high_curr = q_vals[0], q_vals[1]
        if self.quantile_smooth > 0.0 and self._low_ema is not None:
            qs = self.quantile_smooth
            low = (1.0 - qs) * low_curr + qs * self._low_ema
            high = (1.0 - qs) * high_curr + qs * self._high_ema
        else:
            low, high = low_curr, high_curr
        self._low_ema = low.detach()
        self._high_ema = high.detach()

        denom = (high - low).clamp_min(1e-6)
        depth_norm = ((depth_raw - low) / denom).clamp(0.0, 1.0)
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "stab_quantile_ema", time.perf_counter() - t0)

        # ---------- 'near' 空间转换 ----------
        t0 = time.perf_counter()
        near_smooth = depth_norm if self.depth_mode == "inverse" else (1.0 - depth_norm)
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "stab_near_calc", time.perf_counter() - t0)

        # ---------- Plan D: 光流对齐 prev_near / prev_rgb ----------
        # （在 EMA 之前完成空间对齐；与 B 协同：D 修对齐、B 修兜底）
        prev_near_aligned = self._prev_near
        prev_rgb_aligned = self._prev_rgb
        if (
                self.flow_align_enabled
                and self._prev_near is not None
                and self._prev_rgb is not None
                and self.flow_estimator is not None
        ):
            t0 = time.perf_counter()
            flow = self.flow_estimator.estimate_backward_flow(
                rgb_depth, target_size_hw=tuple(near_smooth.shape),
            )
            if flow is not None:
                prev_near_aligned = backward_warp(self._prev_near, flow)
                prev_rgb_aligned = backward_warp(self._prev_rgb, flow)
            _maybe_sync(device, profile_sync)
            _stage_add(stage_times, "stab_flow_align", time.perf_counter() - t0)

        # ---------- Plan B + C: RGB 引导的逐像素 EMA ----------
        if self.depth_smooth > 0.0 and prev_near_aligned is not None:
            t0 = time.perf_counter()
            if self.rgb_motion_sigma > 0.0 and prev_rgb_aligned is not None:
                # 注意：若启用 D，此处 prev_rgb_aligned 是 warp 后的，
                #      diff 反映 "对齐后的残差"，flow 估对处 diff 小→更强平滑。
                diff_sq = (rgb_depth - prev_rgb_aligned).pow(2).sum(dim=0)  # [H, W]
                motion = 1.0 - torch.exp(-diff_sq / (self.rgb_motion_sigma ** 2))
                a_curr = 1.0 - self.depth_smooth * (1.0 - motion)
            else:
                a_curr = 1.0 - self.depth_smooth
            near_smooth = a_curr * near_smooth + (1.0 - a_curr) * prev_near_aligned
            _maybe_sync(device, profile_sync)
            _stage_add(stage_times, "stab_rgb_adaptive_ema", time.perf_counter() - t0)

        # ---------- 更新跨帧状态（保存 post-EMA 但 pre-median） ----------
        # （median 不进入 prev_near，避免反复叠加）
        self._prev_near = near_smooth.detach().clone()
        self._prev_rgb = rgb_depth.detach().clone()

        # ---------- Plan E: 滑窗逐像素中值 ----------
        if self.median_window > 1:
            t0 = time.perf_counter()
            self._near_history.append(near_smooth.detach())
            if len(self._near_history) >= 2:
                stack = torch.stack(list(self._near_history), dim=0)  # [K, H, W]
                near_smooth = stack.median(dim=0).values
            _maybe_sync(device, profile_sync)
            _stage_add(stage_times, "stab_median_window", time.perf_counter() - t0)

        return near_smooth


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
    # ========== int64 无损 Z-buffer 算法 ==========
    NEAR_BITS = 20
    src_bits = max(20, (N - 1).bit_length())  # 坐标需要的 bit 数
    assert NEAR_BITS + src_bits <= 62, "分辨率过高，超出 int64 编码范围"

    # 1. 深度量化到 20 bit（0~1048575）
    near_q = (near_flat.clamp(0.0, 1.0) * ((1 << NEAR_BITS) - 1)).long()
    # 2. 位拼接：深度放在高位，坐标放在低位
    encoded = (near_q << src_bits) | src_lin.long()

    # 3. scatter_reduce_ 找每个目标位置的最大值
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
    out[:, -dilate_px:] = hole[:, -dilate_px:]  # 最右侧边界还原
    return out


@torch.no_grad()
def fast_inpaint_gpu(
    image: torch.Tensor, hole_mask: torch.Tensor, kernel_size: int, max_iter: int,
    stage_times: Dict[str, float], profile_sync: bool,
    near: Optional[torch.Tensor] = None, bg_beta: float = 6.0,
) -> torch.Tensor:
    """背景优先补洞：near 越小(越远=背景)权重越大，避免前景色污染 disocclusion。"""
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
    kernel1, kernel3 = _get_inpaint_kernels(
        kernel_size, str(device), str(img.dtype).split('.')[-1]
    )

    # 背景权重：near 小(远)→权重大。near 缺省时退化为均匀权重(=原行为)
    if near is not None:
        bg_w = torch.exp(-bg_beta * near.clamp(0, 1)).to(img.dtype)  # [H,W]
        bg_w = bg_w.unsqueeze(0).unsqueeze(0)                        # [1,1,H,W]
    else:
        bg_w = torch.ones((1, 1, *hole.shape), device=device, dtype=img.dtype)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_init_kernel", time.perf_counter() - t0)

    t0 = time.perf_counter()
    iter_count = 0
    for _ in range(max_iter):
        iter_count += 1
        if torch.count_nonzero(hole) == 0:
            break
        known = (~hole).float().unsqueeze(0).unsqueeze(0)
        w = known * bg_w                       # 只用已知像素，且偏向背景
        img_nchw = img.permute(2, 0, 1).unsqueeze(0)
        rgb_sum = F.conv2d(img_nchw * w, kernel3, padding=pad, groups=3)
        count = F.conv2d(w, kernel1, padding=pad).clamp_min(1e-6)
        avg = rgb_sum / count
        fillable = hole & (F.conv2d(known, kernel1, padding=pad)[0, 0] > 0)
        if not torch.any(fillable):
            break
        avg_hwc = avg[0].permute(1, 2, 0)
        img[fillable] = avg_hwc[fillable]
        hole[fillable] = False
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, f"inpaint_conv_{iter_count}iter", time.perf_counter() - t0)

    # ★ 兜底修复：剩余洞不要填回黑色，用全图已知区域均值(近似背景)填
    t0 = time.perf_counter()
    if torch.any(hole):
        known_pixels = image[~hole_mask]            # 原始已知像素
        fallback = known_pixels.mean(dim=0) if known_pixels.numel() > 0 \
                   else torch.zeros(3, device=device, dtype=img.dtype)
        img[hole] = fallback
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_final_fill_remaining", time.perf_counter() - t0)
    return img


def compute_aspect_preserved_size(orig_h: int, orig_w: int, long_edge: int) -> Tuple[int, int]:
    """保持宽高比，计算目标尺寸（确保两个维度都是 14 的倍数）"""
    scale = long_edge / max(orig_h, orig_w)
    new_h = max(14, int(round(orig_h * scale / 14)) * 14)
    new_w = max(14, int(round(orig_w * scale / 14)) * 14)
    if orig_w >= orig_h:
        new_w = long_edge
    else:
        new_h = long_edge
    return new_h, new_w


def compose_stereo(left_u8: np.ndarray, right_u8: np.ndarray, layout: str, overlay_alpha: float) -> np.ndarray:
    if layout == "sbs":
        return np.concatenate([left_u8, right_u8], axis=1)
    if layout == "ou":
        return np.concatenate([left_u8, right_u8], axis=0)
    if layout == "anaglyph":
        # 红青3D：左眼看红色通道，右眼看青(绿+蓝)通道
        result = np.zeros_like(left_u8)
        result[..., 0] = left_u8[..., 0]
        result[..., 1] = right_u8[..., 1]
        result[..., 2] = right_u8[..., 2]
        return result
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    mixed = (1.0 - a) * left_u8.astype(np.float32) + a * right_u8.astype(np.float32)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def check_nvenc_available(ffmpeg_bin: str, test_width: int = 1920, test_height: int = 1080) -> bool:
    """测试 h264_nvenc 是否可用"""
    try:
        # 用一个简单的测试命令
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={test_width}x{test_height}:r=1",
            "-vframes", "1",
            "-c:v", "h264_nvenc", "-gpu", "0",
            "-f", "null", "/dev/null"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def create_nvenc_writer(
    ffmpeg_bin: str, input_video: str, output_video: str,
    fps: float, out_w: int, out_h: int, encoder: str, nvenc_gpu: int, preset: str, cq: int,
) -> subprocess.Popen:
    # yuv420p 要求宽高为偶数，如果是奇数就用 yuv444p
    pix_fmt = "yuv420p" if (out_w % 2 == 0 and out_h % 2 == 0) else "yuv444p"
    if pix_fmt != "yuv420p":
        print(f"[mono2stereo] ⚠️  分辨率 {out_w}×{out_h} 有奇数，使用 {pix_fmt}")

    def build_cmd(enc: str) -> list:
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-", "-i", input_video,
            "-map", "0:v:0", "-map", "1:a?",
        ]
        if enc == "h264_nvenc":
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-gpu", str(nvenc_gpu),
                "-preset", preset,
                "-rc", "vbr",
                "-cq", str(cq),
                "-b:v", "0",
            ])
        elif enc == "libx264":
            cmd.extend([
                "-c:v", "libx264",
                "-preset", preset if preset in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"] else "medium",
                "-crf", str(cq),
            ])
        elif enc == "h264_vaapi":
            cmd.extend([
                "-c:v", "h264_vaapi",
                "-qp", str(cq),
            ])
        cmd.extend([
            "-pix_fmt", pix_fmt,
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-shortest",
            output_video,
        ])
        return cmd

    # 简化版本：先试 h264_nvenc，如果报错信息里有分辨率相关的，再考虑处理
    # 这里不做复杂的预检测，直接启动，如果后续写入失败，main函数已经有处理逻辑了
    actual_encoder = encoder
    cmd = build_cmd(actual_encoder)
    print(f"[mono2stereo] ffmpeg (using {actual_encoder}):", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


# ==================== 全 GPU 双分辨率预处理 ====================
@torch.no_grad()
def prepare_inputs_dual_res_gpu(
    frame_bgr: np.ndarray,
    device: torch.device,
    depth_size_hw: Tuple[int, int],
    dibr_size_hw: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    stage_times: Dict[str, float],
    profile_sync: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    一次上传同时生成三件套：
        model_input         [1,3,depth_h,depth_w]   ImageNet 标准化的模型输入
        left_rgb_dibr_hwc   [dibr_h,dibr_w,3]       DIBR 渲染底图 [0,1]
        rgb_depth_chw       [3,depth_h,depth_w]     时序稳定器用的同分辨率 RGB [0,1]
    """
    t0 = time.perf_counter()
    bgr_hwc = torch.from_numpy(frame_bgr).to(
        device=device, dtype=torch.float32, non_blocking=True,
    )
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_h2d", time.perf_counter() - t0)

    t0 = time.perf_counter()
    rgb_nchw = bgr_hwc.flip(-1).permute(2, 0, 1).unsqueeze(0) / 255.0

    # 分支 1：深度推理分辨率（同时给深度模型和时序稳定器使用）
    rgb_depth_nchw = F.interpolate(rgb_nchw, depth_size_hw, mode="bilinear", align_corners=False)

    # 分支 2：DIBR 渲染分辨率
    if dibr_size_hw == (rgb_nchw.shape[2], rgb_nchw.shape[3]):
        rgb_dibr_nchw = rgb_nchw
    else:
        rgb_dibr_nchw = F.interpolate(rgb_nchw, dibr_size_hw, mode="bilinear", align_corners=False)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_color_and_resize_gpu", time.perf_counter() - t0)

    t0 = time.perf_counter()
    # 给稳定器用的未标准化同分辨率 RGB
    rgb_depth_chw = rgb_depth_nchw[0].contiguous()

    # 给深度模型用的 ImageNet 标准化输入
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    model_input = (rgb_depth_nchw - mean_t) / std_t

    # DIBR HWC 底图
    left_rgb_dibr_hwc = rgb_dibr_nchw[0].permute(1, 2, 0).contiguous()
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_normalize", time.perf_counter() - t0)

    return model_input, left_rgb_dibr_hwc, rgb_depth_chw


# ==================== 视频帧预读队列 ====================
class FrameReaderThread(threading.Thread):
    """后台线程预读视频帧，放到队列里让 CPU 和 GPU 并行工作"""
    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 8):
        super().__init__(daemon=True)
        self.cap = cap
        self.queue: "queue.Queue[Tuple[bool, Optional[np.ndarray]]]" = queue.Queue(maxsize=queue_size)
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


def main() -> None:
    script_t0 = time.perf_counter()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("该脚本要求CUDA，享受全GPU加速。")

    torch.backends.cudnn.benchmark = True

    # ==================== 双模式支持 ====================
    if args.video_model:
        # Video-Depth-Anything 视频模型（内置时序一致性，官方流式推理）
        if not VIDEO_MODEL_AVAILABLE:
            raise ImportError(
                "\n" + "="*70 + "\n"
                "Video-Depth-Anything 导入失败！\n"
                "请确认 submodules/Video_Depth_Anything 目录存在\n"
                "或者先不加 --video-model，用单帧 + 时序稳定A-E模式\n"
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
        # 单帧模型 + PLAN A-E 五重时序稳定
        model = DepthAnythingV2(**get_model_config(args.encoder))
        ckpt = args.ckpt or f"checkpoints/depth_anything_v2_{args.encoder}.pth"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        if args.fp16:
            model = model.half()
        print(f"[mono2stereo] ✅ 使用单帧模型 + PLAN A-E 五重时序稳定")

    # 预处理参数
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # -------------------- 时序稳定器（仅单帧模式使用） --------------------
    stabilizer = None
    if not args.video_model:
        stabilizer = TemporalDepthStabilizer(
            depth_mode=args.depth_mode,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            depth_smooth=args.depth_smooth,
            quantile_smooth=args.quantile_smooth,
            rgb_motion_sigma=args.rgb_motion_sigma,
            flow_align_enabled=args.flow_align,
            flow_height=args.flow_height,
            median_window=args.median_window,
        )
        print(
            f"[mono2stereo] 🌟 时序稳定器配置: "
            f"quantile_smooth={args.quantile_smooth} (A), "
            f"depth_smooth={args.depth_smooth} (B+C), "
            f"rgb_motion_sigma={args.rgb_motion_sigma} (B), "
            f"flow_align={'ON' if args.flow_align else 'OFF'}"
            f"{f' (h={args.flow_height})' if args.flow_align else ''} (D), "
            f"median_window={args.median_window} (E)"
        )

    # 深度感知补洞提示
    print(f"[mono2stereo] 🎨 深度感知空洞修补: 启用 (bg_beta={args.bg_beta})")

    files = collect_video_files(args.video_path)
    if not files:
        raise FileNotFoundError(f"未找到可处理视频: {args.video_path}")

    is_single_file = len(files) == 1 and args.output != "./output.mp4"
    os.makedirs(args.outdir, exist_ok=True)
    all_processed_frames = 0
    warmup_done = False

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
        aspect_ratio = w_orig / h_orig

        # ---- 深度推理流尺寸 (14 倍数) ----
        # 和 mono2stereo_video_better.py 一致：input_size 是长边
        long_edge = args.input_size
        assert long_edge % 14 == 0, f"input_size({long_edge}) 必须是 14 的倍数"
        scale = long_edge / max(h_orig, w_orig)
        depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
        depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
        if w_orig >= h_orig:
            depth_w = long_edge
        else:
            depth_h = long_edge

        # ---- DIBR 渲染流尺寸 ----
        if args.dibr_size == -1:
            dibr_h, dibr_w = h_orig, w_orig
        elif args.dibr_size == 0:
            dibr_h, dibr_w = depth_h, depth_w  # 和 input-size 一致
        else:
            dibr_h = args.dibr_size
            dibr_w = int(round(dibr_h * aspect_ratio / 2.0)) * 2

        # 视差像素跨分辨率归一化
        max_disparity_dibr = args.max_disparity * dibr_w / w_orig

        print(f"[mono2stereo] original resolution : {w_orig}×{h_orig}, FPS={fps_orig:.1f}")
        print(f"[mono2stereo] Depth AI resolution : {depth_w}×{depth_h} (严格 14 倍数)")
        print(f"[mono2stereo] DIBR Core resolution: {dibr_w}×{dibr_h}")
        print(f"[mono2stereo] max-disparity (orig): {args.max_disparity:.1f} → (DIBR-res): {max_disparity_dibr:.2f}")
        print(f"[mono2stereo] 🚀 int64无损DIBR + 前景边缘保护 (hole-dilate-left={args.hole_dilate_left})")

        # Warmup（仅单帧模式）
        if args.warmup_iters > 0 and not args.video_model and not warmup_done:
            print("[mono2stereo] CUDA Warmup 开始...")
            warmup_dtype = torch.float16 if args.fp16 else torch.float32
            dummy = torch.randn(1, 3, depth_h, depth_w, device=device, dtype=warmup_dtype)
            for _ in range(args.warmup_iters):
                _ = model(dummy)
            torch.cuda.synchronize()
            warmup_done = True

        # 输出尺寸 - 自动适配 NVENC
        use_layout = args.layout
        if args.video_encoder == "h264_nvenc":
            # 如果是 h264_nvenc，先试试 sbs 宽度是否超限制
            sbs_w = w_orig * 2
            nvenc_max_w = 4096  # 经过测试，当前环境 NVENC 最大宽度支持超过 4096，但 5152 不行
            if use_layout == "sbs" and sbs_w > nvenc_max_w:
                print(f"[mono2stereo] ⚠️  sbs 模式宽度 {sbs_w} 超过 NVENC 限制，自动切换到 ou 模式")
                use_layout = "ou"

        # 计算最终输出尺寸
        if use_layout == "sbs":
            out_w, out_h = w_orig * 2, h_orig
        elif use_layout == "ou":
            out_w, out_h = w_orig, h_orig * 2
        else:
            out_w, out_h = w_orig, h_orig

        if is_single_file:
            out_path = args.output
        else:
            out_path = os.path.join(args.outdir, f"{Path(filename).stem}_3d.mp4")

        writer = create_nvenc_writer(
            ffmpeg_bin=args.ffmpeg_bin, input_video=filename, output_video=out_path,
            fps=fps_orig, out_w=out_w, out_h=out_h,
            encoder=args.video_encoder, nvenc_gpu=args.nvenc_gpu,
            preset=args.nvenc_preset, cq=args.nvenc_cq,
        )
        if writer.stdin is None:
            cap.release()
            raise RuntimeError("ffmpeg管道初始化失败")

        stage_times: Dict[str, float] = {}
        processed_frames = 0
        t_video0 = time.perf_counter()

        # 每个新视频清空稳定器状态
        if stabilizer is not None:
            stabilizer.reset()

        # 启动后台预读线程
        reader = FrameReaderThread(cap, queue_size=args.queue_size)
        reader.start()

        try:
            while True:
                frame_t0 = time.perf_counter()

                # 从队列拿帧
                t0 = time.perf_counter()
                ok, frame_bgr = reader.get_frame()
                _stage_add(stage_times, "read_frame_queue", time.perf_counter() - t0)
                if not ok:
                    break

                # ========== Step 1: 深度推理 ==========
                t0 = time.perf_counter()

                if args.video_model:
                    # 视频模型模式：官方流式推理
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                        depth_raw_np = model.infer_video_depth_one(
                            frame_rgb, input_size=args.input_size, device=device, fp32=(not args.fp16)
                        )
                    depth_raw = torch.from_numpy(depth_raw_np).to(device=device, dtype=torch.float32)
                    _stage_add(stage_times, "model_inference_video", time.perf_counter() - t0)

                    # 视频模式：DIBR 用原图分辨率
                    left_rgb_dibr = torch.from_numpy(frame_bgr).to(device=device, dtype=torch.float32)
                    left_rgb_dibr = left_rgb_dibr.flip(-1) / 255.0  # BGR→RGB
                    near_smooth = depth_raw  # 视频模型自带时序，直接用
                else:
                    # 单帧模式预处理
                    use_stabilizer = args.quantile_smooth > 0 or args.depth_smooth > 0 or args.flow_align or args.median_window > 1
                    need_two_res = (dibr_h != depth_h or dibr_w != depth_w) or use_stabilizer
                    if need_two_res:
                        model_input, left_rgb_dibr, rgb_depth = prepare_inputs_dual_res_gpu(
                            frame_bgr, device, (depth_h, depth_w), (dibr_h, dibr_w),
                            MEAN, STD, stage_times, args.profile_time,
                        )
                    else:
                        # 时序稳定都禁用且 dibr-size == input-size，用简化预处理
                        model_input, left_rgb_dibr, _ = prepare_inputs_dual_res_gpu(
                            frame_bgr, device, (depth_h, depth_w), (depth_h, depth_w),
                            MEAN, STD, stage_times, args.profile_time,
                        )

                    if args.fp16:
                        model_input = model_input.half()

                    depth_raw = infer_depth_lowres(
                        model, model_input, args.fp16, stage_times, args.profile_time
                    )
                    _stage_add(stage_times, "infer_depth_total", time.perf_counter() - t0)

                    # 累加预处理总耗时
                    _stage_add(stage_times, "preprocess_total",
                        stage_times.get("prep_h2d", 0) +
                        stage_times.get("prep_color_and_resize_gpu", 0) +
                        stage_times.get("prep_normalize", 0)
                    )

                    # ========== Step 2: 深度归一化 + near 转换 ==========
                    t0 = time.perf_counter()
                    if use_stabilizer:
                        near_smooth = stabilizer.step(depth_raw, rgb_depth, stage_times, args.profile_time)
                        _stage_add(stage_times, "stabilize_total", time.perf_counter() - t0)
                    else:
                        # 时序稳定都禁用时，直接做简单归一化（和 mono2stereo_video_better.py 一致）
                        flat = depth_raw.reshape(-1)
                        sample_size = 16384
                        idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
                        sample = flat[idx]
                        q_vals = torch.quantile(sample, torch.tensor([args.clip_low, args.clip_high], device=flat.device))
                        low, high = q_vals[0], q_vals[1]
                        denom = (high - low).clamp_min(1e-6)
                        depth_norm = ((depth_raw - low) / denom).clamp(0.0, 1.0)
                        near_smooth = depth_norm if args.depth_mode == "inverse" else (1.0 - depth_norm)
                        _maybe_sync(device, args.profile_time)
                        _stage_add(stage_times, "stabilize_total", time.perf_counter() - t0)

                if args.profile_time:
                    torch.cuda.synchronize()

                # ========== Step 3: near → disparity ==========
                t0 = time.perf_counter()
                if args.video_model:
                    # 视频模式：简单处理
                    if args.depth_mode == "inverse":
                        near = depth_raw
                    else:
                        near = 1.0 - depth_raw
                    disparity = near * args.max_disparity
                    # 视频模式下，depth_raw 和 left_rgb_dibr 都是原分辨率
                    near_for_inpaint = near
                else:
                    # 单帧模式：GPU重采样到 DIBR 分辨率（如果需要）
                    if (dibr_h == depth_h and dibr_w == depth_w):
                        near_dibr = near_smooth
                    else:
                        near_dibr = F.interpolate(
                            near_smooth[None, None, :, :], size=(dibr_h, dibr_w),
                            mode="bilinear", align_corners=False,
                        )[0, 0]
                    disparity = near_dibr * max_disparity_dibr
                    near_for_inpaint = near_dibr
                    _maybe_sync(device, args.profile_time)
                    _stage_add(stage_times, "near_resample_gpu", time.perf_counter() - t0)

                _stage_add(stage_times, "near_to_disparity", time.perf_counter() - t0)

                # ========== Step 4: GPU DIBR ==========
                t0 = time.perf_counter()
                right_rgb_dibr, hole_dibr = forward_warp_right_gpu(
                    left_rgb_dibr, disparity,
                    near_for_inpaint,  # 传递 near score 给 DIBR（用于 Z-buffer）
                    stage_times, args.profile_time
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "gpu_warp_total", time.perf_counter() - t0)

                # ========== 空洞左侧膨胀 ==========
                if args.hole_dilate_left > 0:
                    t0 = time.perf_counter()
                    hole_dilated = dilate_hole_left(hole_dibr, args.hole_dilate_left)
                    if args.profile_time:
                        torch.cuda.synchronize()
                    _stage_add(stage_times, "hole_dilate_left", time.perf_counter() - t0)
                else:
                    hole_dilated = hole_dibr

                # ========== Step 5: GPU 快速补洞（深度感知版）==========
                t0 = time.perf_counter()
                right_inpainted_dibr = fast_inpaint_gpu(
                    right_rgb_dibr, hole_mask=hole_dilated,
                    kernel_size=args.fast_kernel, max_iter=args.fast_max_iter,
                    stage_times=stage_times, profile_sync=args.profile_time,
                    near=near_for_inpaint,  # 传递深度信息
                    bg_beta=args.bg_beta,  # 背景偏好系数
                )
                if args.profile_time:
                    torch.cuda.synchronize()
                _stage_add(stage_times, "fast_inpaint_total", time.perf_counter() - t0)

                # ========== Step 6: GPU → CPU + 上采样 ==========
                t0 = time.perf_counter()
                right_u8_dibr = (right_inpainted_dibr.clamp(0, 1) * 255.0).byte().contiguous().cpu().numpy()
                _stage_add(stage_times, "to_cpu_numpy", time.perf_counter() - t0)

                t0 = time.perf_counter()
                # 左眼直接用原图（避免下采样再上采样的画质损失）
                left_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # 右眼才需要上采样
                if args.video_model or (dibr_h == h_orig and dibr_w == w_orig):
                    right_u8 = right_u8_dibr
                else:
                    right_u8 = cv2.resize(right_u8_dibr, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                _stage_add(stage_times, "upsample_to_orig_cpu", time.perf_counter() - t0)

                t0 = time.perf_counter()
                stereo = compose_stereo(left_u8, right_u8, use_layout, args.overlay_alpha)
                _stage_add(stage_times, "compose_stereo", time.perf_counter() - t0)

                # ========== Step 7: 写入 ffmpeg ==========
                t0 = time.perf_counter()
                writer.stdin.write(stereo.tobytes())
                _stage_add(stage_times, "write_to_ffmpeg", time.perf_counter() - t0)

                _stage_add(stage_times, "total_per_frame", time.perf_counter() - frame_t0)
                processed_frames += 1
                all_processed_frames += 1
                if processed_frames % 30 == 0:
                    print(f"[mono2stereo] {Path(filename).stem} rendered {processed_frames} frames")
        finally:
            reader.stop()
            cap.release()
            writer.stdin.close()
            # 读取stderr看报错信息
            stderr_output = writer.stderr.read().decode('utf-8', errors='replace')
            ret = writer.wait()
            if ret != 0:
                if stderr_output:
                    print(f"\n[mono2stereo] ❌ ffmpeg 错误信息:\n{stderr_output}")
                raise RuntimeError(f"ffmpeg编码失败，退出码: {ret}")

        total_elapsed = time.perf_counter() - t_video0
        avg_fps = processed_frames / max(total_elapsed, 1e-6)
        print(f"[mono2stereo] output={out_path}")
        print(f"[mono2stereo] processed_frames={processed_frames}, total_time={total_elapsed:.3f}s, avg_fps={avg_fps:.3f}")

        if args.profile_time and processed_frames > 0:
            # 收集所有统计信息
            profile_lines = []
            profile_lines.append("\n[mono2stereo] 各阶段用时统计:")
            profile_lines.append("=" * 70)
            profile_lines.append(f"📈 处理帧数: {processed_frames}, 总用时: {total_elapsed:.3f}s, 平均FPS: {processed_frames / total_elapsed:.2f}")
            profile_lines.append("")

            top_level = [
                ("read_frame_queue", "从队列拿帧 (后台预读)"),
                ("preprocess_total", "⚡ 输入预处理总耗时"),
                ("infer_depth_total", "⚡ 深度推理总耗时"),
                ("stabilize_total", "🌟 时序稳定总耗时 (Plan A+B+C+D+E)"),
                ("near_resample_gpu", "⚡ near空间GPU双线性重采样到DIBR分辨率"),
                ("near_to_disparity", "near → disparity"),
                ("gpu_warp_total", "⚡ DIBR图像扭曲总耗时 (int64)"),
                ("hole_dilate_left", "空洞左侧膨胀 (保护前景)"),
                ("fast_inpaint_total", "⚡ 快速补洞总耗时 (深度感知版)"),
                ("to_cpu_numpy", "结果转回CPU"),
                ("upsample_to_orig_cpu", "上采样回原分辨率 (CPU)"),
                ("compose_stereo", "立体帧拼接合成"),
                ("write_to_ffmpeg", "写入ffmpeg编码"),
            ]

            top_total = sum(stage_times.get(k, 0) for k, _ in top_level)
            profile_lines.append(f"📊 【顶层阶段汇总】 (总和: {top_total:.3f}s, 平均每帧: {top_total/processed_frames*1000:.1f}ms)")
            for key, cn_name in top_level:
                if key in stage_times:
                    t = stage_times[key]
                    t_avg = t / processed_frames * 1000
                    ratio = (t / top_total * 100.0) if top_total > 0 else 0.0
                    profile_lines.append(f"  {key:30s} {t:8.3f}s ({ratio:5.1f}%) | {t_avg:7.1f}ms/帧 | {cn_name}")

            profile_lines.append("\n🔍 【各阶段内部细分明细】")
            profile_lines.append("-" * 70)

            # 预处理内部
            depth_sub = [
                ("prep_h2d", "原图传到GPU"),
                ("prep_color_and_resize_gpu", "BGR转RGB + 双分辨率Resize (全GPU)"),
                ("prep_normalize", "Normalize标准化 (GPU)"),
                ("model_inference", "⭐ DepthAnything模型推理 (单帧)"),
                ("model_inference_video", "⭐ DepthAnything模型推理 (视频多帧)"),
            ]
            depth_parent = "infer_depth_total"
            if depth_parent in stage_times:
                parent_time = stage_times[depth_parent]
                parent_avg = parent_time / processed_frames * 1000
                profile_lines.append(f"\n[{depth_parent}] 内部拆解 (总: {parent_time:.3f}s, 平均: {parent_avg:.1f}ms/帧):")
                for key, cn_name in depth_sub:
                    if key in stage_times:
                        t = stage_times[key]
                        t_avg = t / processed_frames * 1000
                        ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                        profile_lines.append(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%) | {t_avg:6.1f}ms/帧 | {cn_name}")

            # 时序稳定内部
            if "stabilize_total" in stage_times:
                parent_time = stage_times["stabilize_total"]
                parent_avg = parent_time / processed_frames * 1000
                profile_lines.append(f"\n[stabilize_total] 内部拆解 (总: {parent_time:.3f}s, 平均: {parent_avg:.1f}ms/帧):")
                stab_sub = [
                    ("stab_quantile_ema", "🅰 分位数EMA + 归一化"),
                    ("stab_near_calc", "near转换 (1-d / d)"),
                    ("stab_flow_align", "🅳 光流对齐 (Farneback + warp)"),
                    ("stab_rgb_adaptive_ema", "🅱 RGB引导自适应EMA"),
                    ("stab_median_window", "🅴 滑窗逐像素中值"),
                ]
                for key, cn_name in stab_sub:
                    if key in stage_times:
                        t = stage_times[key]
                        t_avg = t / processed_frames * 1000
                        ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                        profile_lines.append(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%) | {t_avg:6.1f}ms/帧 | {cn_name}")

            # DIBR内部
            warp_parent = "gpu_warp_total"
            if warp_parent in stage_times:
                parent_time = stage_times[warp_parent]
                parent_avg = parent_time / processed_frames * 1000
                profile_lines.append(f"\n[{warp_parent}] 内部拆解 (总: {parent_time:.3f}s, 平均: {parent_avg:.1f}ms/帧):")
                warp_sub = [
                    ("warp_grid_gen", "生成网格坐标"),
                    ("warp_index_prep", "准备线性索引"),
                    ("warp_z_buffer", "Z-buffer冲突解决 (int64无损)"),
                    ("warp_scatter_pixels", "像素散射"),
                ]
                for key, cn_name in warp_sub:
                    if key in stage_times:
                        t = stage_times[key]
                        t_avg = t / processed_frames * 1000
                        ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                        profile_lines.append(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%) | {t_avg:6.1f}ms/帧 | {cn_name}")

            # 补洞内部
            if "fast_inpaint_total" in stage_times:
                parent_time = stage_times["fast_inpaint_total"]
                parent_avg = parent_time / processed_frames * 1000
                profile_lines.append(f"\n[fast_inpaint_total] 内部拆解 (总: {parent_time:.3f}s, 平均: {parent_avg:.1f}ms/帧):")
                inpaint_keys = [k for k in stage_times if k.startswith("inpaint_")]
                for key in sorted(inpaint_keys):
                    t = stage_times[key]
                    t_avg = t / processed_frames * 1000
                    ratio = (t / parent_time * 100.0) if parent_time > 0 else 0.0
                    if key == "inpaint_skip_no_holes":
                        cn_name = "跳过 (无空洞)"
                    elif key == "inpaint_init_kernel":
                        cn_name = "初始化卷积核"
                    elif key == "inpaint_final_fill_remaining":
                        cn_name = "剩余空洞兜底"
                    elif key.startswith("inpaint_conv_"):
                        iters = key.replace("inpaint_conv_", "").replace("iter", "")
                        cn_name = f"卷积迭代 ({iters}次) [深度感知]"
                    else:
                        cn_name = ""
                    profile_lines.append(f"  ├─ {key:25s} {t:8.3f}s ({ratio:5.1f}%) | {t_avg:6.1f}ms/帧 | {cn_name}")

            profile_lines.append("")
            profile_lines.append("=" * 70)
            profile_lines.append(f"💡 提示: 如果 'read_frame_queue' 接近 0，说明预读队列工作正常，GPU 没有等 CPU！")

            # 打印到控制台
            for line in profile_lines:
                print(line)

            # 保存到文件（如果指定了路径或默认保存）
            profile_file = args.profile_output
            if profile_file is None:
                # 默认保存在输出视频同目录下，同名 .txt 文件
                profile_file = Path(out_path).with_suffix('.txt')

            try:
                os.makedirs(os.path.dirname(profile_file), exist_ok=True)
                with open(profile_file, 'w', encoding='utf-8') as f:
                    # 写入基本信息
                    f.write(f"=== 2D转3D 性能统计 (深度感知补洞版) ===\n")
                    f.write(f"视频文件: {filename}\n")
                    f.write(f"输出文件: {out_path}\n")
                    f.write(f"原始分辨率: {w_orig}×{h_orig}\n")
                    f.write(f"深度推理分辨率: {depth_w}×{depth_h}\n")
                    f.write(f"DIBR分辨率: {dibr_w}×{dibr_h}\n")
                    f.write(f"编码器: {args.encoder}\n")
                    f.write(f"输入尺寸: {args.input_size}\n")
                    f.write(f"最大视差: {args.max_disparity}\n")
                    f.write(f"FP16: {args.fp16}\n")
                    f.write(f"深度感知补洞: 是\n")
                    f.write(f"bg_beta: {args.bg_beta}\n")
                    f.write(f"\n=== Plan配置 ===\n")
                    f.write(f"quantile_smooth (Plan A): {args.quantile_smooth}\n")
                    f.write(f"depth_smooth (Plan B+C): {args.depth_smooth}\n")
                    f.write(f"rgb_motion_sigma (Plan B): {args.rgb_motion_sigma}\n")
                    f.write(f"flow_align (Plan D): {args.flow_align}\n")
                    f.write(f"flow_height (Plan D): {args.flow_height}\n")
                    f.write(f"median_window (Plan E): {args.median_window}\n")
                    f.write(f"\n=== 运行结果 ===\n")
                    f.write(f"处理帧数: {processed_frames}\n")
                    f.write(f"总用时: {total_elapsed:.3f}s\n")
                    f.write(f"平均FPS: {processed_frames / total_elapsed:.2f}\n")
                    f.write(f"\n=== 详细统计 ===\n")
                    for line in profile_lines:
                        f.write(line + '\n')
                print(f"[mono2stereo] 性能统计已保存到: {profile_file}")
            except Exception as e:
                print(f"[mono2stereo] 保存性能统计失败: {e}")

    if args.profile_total:
        script_elapsed = time.perf_counter() - script_t0
        script_avg_fps = all_processed_frames / max(script_elapsed, 1e-6)
        print(f"\n[mono2stereo][total] frames={all_processed_frames}, avg_fps={script_avg_fps:.3f}")


if __name__ == "__main__":
    main()
