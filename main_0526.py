# python main_0526.py --video-path /mnt/A/hust_zhang/Project/Simulation/2Dto3D_past1/data/video-test/video2.mp4 \
# --output ./output/video2_vda.mp4 --profile-time --profile-total

import argparse
import functools
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import subprocess
import sys
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

_vda_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submodules', 'Video_Depth_Anything')
if _vda_root not in sys.path:
    sys.path.insert(0, _vda_root)
from video_depth_anything.video_depth_stream import VideoDepthAnything


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D转3D: 双流分辨率 + 时序稳定 (A+B+C+D+E) + NVENC"
    )
    parser.add_argument("--video-path", type=str, default="./video.mp4",
                        help="输入视频路径、视频目录或 txt 列表")
    parser.add_argument("--output", type=str, default="./output.mp4",
                        help="输出视频路径（单文件模式）")
    parser.add_argument("--outdir", type=str, default="./vis_video_3d",
                        help="输出目录（批量模式）")

    # ---- 分辨率 ----
    parser.add_argument("--input-size", type=int, default=1400,  # 252
                        help="深度模型输入基准高度（必须为 14 的倍数，决定推理速度）")
    parser.add_argument("--dibr-size", type=int, default=-1,
                        help="DIBR 渲染基准高度（-1 表示使用视频原分辨率）")

    # ---- 模型 ----
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--ckpt", type=str,
                        default="./submodules/depth/vda/checkpoints/video_depth_anything_vits.pth",
                        help="模型权重路径")
    parser.add_argument("--vda-metric", action="store_true",
                        help="使用 VDA metric depth 模型（默认使用 relative depth）")
    parser.add_argument("--vda-fp32", action="store_true",
                        help="VDA 推理使用 fp32（默认 fp16，更快）")
    parser.add_argument("--warmup-iters", type=int, default=0, help="CUDA warmup 轮数（VDA 流式无需 warmup）")
    parser.add_argument("--queue-size", type=int, default=8, help="预读队列大小")

    # ---- 视差 ----
    parser.add_argument("--max-disparity", type=float, default=16,
                        help="原分辨率等效最大水平视差像素，内部自动按 DIBR 尺寸缩放")
    parser.add_argument("--depth-mode", choices=["metric", "inverse"], default="inverse",
                        help="metric: 值小更近；inverse: 值大更近")
    parser.add_argument("--clip-low", type=float, default=0.01, help="深度归一化低分位")
    parser.add_argument("--clip-high", type=float, default=0.99, help="深度归一化高分位")

    # ---- 时序稳定化（Plan A / B / C） ----
    # 所有 *-smooth 的统一语义：前一帧权重 / 平滑强度，0=禁用。
    parser.add_argument("--depth-smooth", type=float, default=0.0,
                        help="【Plan B+C】near 空间 EMA 强度（前帧权重；0=禁用；VDA 已有时序一致性，"
                             "通常不需要；如需额外平滑建议 0.3~0.5）")
    parser.add_argument("--quantile-smooth", type=float, default=0.0,
                        help="【Plan A】分位数 EMA 强度（前帧权重；0=禁用；VDA 已处理尺度稳定性）")
    parser.add_argument("--rgb-motion-sigma", type=float, default=0.0,
                        help="【Plan B】RGB 帧差 sigma；越小越敏感于运动；"
                             "0=禁用自适应（退化为均匀 EMA）")

    # ---- 时序稳定化（Plan D：光流对齐） ----
    parser.add_argument("--flow-align", action="store_true",
                        help="【Plan D】启用 CPU Farneback 光流补偿的 motion-aligned EMA "
                             "（约 +1~3ms/帧，运动剧烈时画质收益明显）")
    parser.add_argument("--flow-height", type=int, default=144,
                        help="【Plan D】光流计算分辨率的基准高度；越大越精确越慢，144 推荐")

    # ---- 时序稳定化（Plan E：滑窗中值） ----
    parser.add_argument("--median-window", type=int, default=1,
                        help="【Plan E】滑窗中值滤波窗口大小（1=禁用；3 推荐；"
                             "最大 7。引入 (K-1)/2 帧延迟，离线无影响）")

    # ---- 补洞 ----
    parser.add_argument("--fast-kernel", type=int, default=7, help="FAST 补洞邻域核大小（奇数）")
    parser.add_argument("--fast-max-iter", type=int, default=64, help="FAST 补洞最大迭代次数")

    # ---- 输出 ----
    parser.add_argument("--layout", choices=["sbs", "ou", "overlay"], default="sbs",
                        help="sbs 并排 / ou 上下 / overlay 重合")
    parser.add_argument("--overlay-alpha", type=float, default=0.5,
                        help="overlay 模式下右眼权重 [0,1]")

    # ---- 编码 ----
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    parser.add_argument("--nvenc-preset", type=str, default="hq")
    parser.add_argument("--nvenc-cq", type=int, default=19)

    # ---- 性能监控 ----
    parser.add_argument("--profile-time", action="store_true",
                        help="输出各阶段平均耗时和 FPS")
    parser.add_argument("--profile-total", action="store_true",
                        help="输出脚本级端到端总耗时")
    return parser.parse_args()


# =============================================================================
# 公共工具
# =============================================================================

def get_model_config(encoder: str) -> Dict:
    return {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }[encoder]


def collect_video_files(video_path: str) -> List[str]:
    p = Path(video_path)
    if p.is_file() and p.suffix.lower() == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f.readlines() if line.strip()]
        return [x for x in files if Path(x).is_file()]
    if p.is_file():
        return [str(p)]
    return sorted(glob.glob(os.path.join(video_path, "**", "*.mp4"), recursive=True))


def _stage_add(stage_times: Dict[str, float], key: str, dt: float) -> None:
    stage_times[key] = stage_times.get(key, 0.0) + dt


def _maybe_sync(device: torch.device, do_sync: bool) -> None:
    if do_sync and device.type == "cuda":
        torch.cuda.synchronize()


# =============================================================================
# 性能优化：缓存 inpaint 卷积核（M-5，避免每帧重建）
# =============================================================================

@functools.lru_cache(maxsize=8)
def _get_inpaint_kernels(kernel_size: int, device_str: str, dtype_str: str):
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
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

        # 后向光流：从 curr 到 prev 的位移场
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
    返回:   与 image 同 shape 的 warped 结果（border 填充）
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
# 时序稳定化模块（Plan A + B + C + D + E 一体）
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
        # M-1: 随机采样 16k 像素算 quantile，误差可忽略，速度提升 ~100×
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
        # M-7: 安全拷贝，避免后续 in-place 改动污染历史状态
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


# =============================================================================
# DIBR 重投影 (int64 无损 Z-buffer)
# =============================================================================

@torch.no_grad()
def forward_warp_right_gpu(
        left_rgb: torch.Tensor,
        disparity: torch.Tensor,
        near_score: torch.Tensor,
        stage_times: Dict[str, float],
        profile_sync: bool,
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


# =============================================================================
# Fast inpaint
# =============================================================================

@torch.no_grad()
def fast_inpaint_gpu(
        image: torch.Tensor,
        hole_mask: torch.Tensor,
        kernel_size: int,
        max_iter: int,
        stage_times: Dict[str, float],
        profile_sync: bool,
) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("fast-kernel 必须是奇数")

    t0 = time.perf_counter()
    img = image.clone()
    hole = hole_mask.clone()
    if not torch.any(hole):
        _stage_add(stage_times, "inpaint_skip_no_holes", time.perf_counter() - t0)
        return img
    device = img.device

    pad = kernel_size // 2
    # M-5: 用缓存的 kernel，避免每帧重建
    kernel1, kernel3 = _get_inpaint_kernels(
        kernel_size, str(device), str(img.dtype).split('.')[-1]
    )
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


# =============================================================================
# 输出 / 编码
# =============================================================================

def compose_stereo(
        left_u8: np.ndarray, right_u8: np.ndarray, layout: str, overlay_alpha: float,
) -> np.ndarray:
    if layout == "sbs":
        return np.concatenate([left_u8, right_u8], axis=1)
    if layout == "ou":
        return np.concatenate([left_u8, right_u8], axis=0)
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    mixed = (1.0 - a) * left_u8.astype(np.float32) + a * right_u8.astype(np.float32)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def create_nvenc_writer(
        ffmpeg_bin: str, input_video: str, output_video: str,
        fps: float, out_w: int, out_h: int, preset: str, cq: int,
) -> subprocess.Popen:
    cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-", "-i", input_video,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "h264_nvenc", "-preset", preset,
        "-rc", "vbr", "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", "-shortest", output_video,
    ]
    print("[mono2stereo] ffmpeg:", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=sys.stderr)


# =============================================================================
# 全 GPU 双分辨率预处理
# =============================================================================

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


# =============================================================================
# 视频帧预读
# =============================================================================

class FrameReaderThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 8):
        super().__init__(daemon=True)
        self.cap = cap
        self.queue: "queue.Queue[Tuple[bool, Optional[np.ndarray]]]" = queue.Queue(
            maxsize=queue_size,
        )
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


# =============================================================================
# 模型推理
# =============================================================================

@torch.no_grad()
def infer_depth_raw_vda(
        model: "VideoDepthAnything",
        frame_bgr: np.ndarray,
        input_size: int,
        device: torch.device,
        fp32: bool,
        depth_h: int,
        depth_w: int,
        stage_times: Dict[str, float],
        profile_sync: bool,
) -> torch.Tensor:
    """VDA 流式单帧推理：接收 BGR numpy 帧 → VDA 推理 → GPU tensor 深度图 [depth_h, depth_w]

    VDA 的 infer_video_depth_one() 内部维护了 32 帧的时序隐藏状态缓存，
    每次调用只处理当前帧，但利用缓存中的历史帧信息做时序 attention，
    实现边读边算的流式时序一致深度估计。
    """
    t0 = time.perf_counter()
    # VDA 内部 transform (NormalizeImage) 期望 RGB [0,1] 输入
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # 流式推理（自动维护时序隐藏状态缓存）
    depth_np = model.infer_video_depth_one(
        frame_rgb, input_size=input_size, device=device, fp32=fp32,
    )
    # depth_np: 原始视频分辨率 [H_orig, W_orig] numpy float32
    # 转到 GPU 并重采样到深度推理分辨率
    depth_t = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    if depth_t.shape[0] != depth_h or depth_t.shape[1] != depth_w:
        depth_t = F.interpolate(
            depth_t[None, None, :, :], size=(depth_h, depth_w),
            mode="bilinear", align_corners=False,
        )[0, 0]
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "model_inference", time.perf_counter() - t0)
    return depth_t


# =============================================================================
# 单视频处理
# =============================================================================

def process_one_video(
        filename: str,
        out_path: str,
        args: argparse.Namespace,
        device: torch.device,
        model: "VideoDepthAnything",
        stabilizer: TemporalDepthStabilizer,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        warmup_state: Dict[str, bool],
) -> Tuple[Dict[str, float], int, float]:
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"[mono2stereo] 跳过，无法打开: {filename}")
        return {}, 0, 0.0

    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    aspect_ratio = w_orig / h_orig

    # ---- 深度推理流尺寸 (14 倍数) ----
    depth_h = args.input_size
    assert depth_h % 14 == 0, f"input_size({depth_h}) 必须是 14 的倍数"
    depth_w = int(round(depth_h * aspect_ratio / 14.0)) * 14

    # ---- DIBR 渲染流尺寸 ----
    if args.dibr_size <= 0:
        dibr_h, dibr_w = h_orig, w_orig
    else:
        dibr_h = args.dibr_size
        dibr_w = int(round(dibr_h * aspect_ratio / 2.0)) * 2

    max_disparity_dibr = args.max_disparity * dibr_w / w_orig

    print(f"[mono2stereo] original resolution : {w_orig}×{h_orig}, FPS={fps_orig:.1f}")
    print(f"[mono2stereo] Depth AI resolution : {depth_w}×{depth_h} (严格 14 倍数)")
    print(f"[mono2stereo] DIBR Core resolution: {dibr_w}×{dibr_h}")
    print(f"[mono2stereo] max-disparity (orig): "
          f"{args.max_disparity:.1f} → (DIBR-res): {max_disparity_dibr:.2f}")

    # ---- 一次性 warmup（VDA 需要 [B,T,C,H,W] 格式的 5D 输入）----
    if args.warmup_iters > 0 and not warmup_state["done"]:
        print("[mono2stereo] CUDA Warmup 开始...")
        dummy = torch.randn(1, 1, 3, depth_h, depth_w, device=device,
                            dtype=torch.float16 if not args.vda_fp32 else torch.float32)
        for _ in range(args.warmup_iters):
            # VDA 的 forward_features 接受 4D [B*T, C, H, W] 输入
            _ = model.forward_features(dummy.flatten(0, 1))
        torch.cuda.synchronize()
        warmup_state["done"] = True

    # ---- 输出尺寸 ----
    if args.layout == "sbs":
        out_w, out_h = w_orig * 2, h_orig
    elif args.layout == "ou":
        out_w, out_h = w_orig, h_orig * 2
    else:
        out_w, out_h = w_orig, h_orig

    writer = create_nvenc_writer(
        ffmpeg_bin=args.ffmpeg_bin, input_video=filename, output_video=out_path,
        fps=fps_orig, out_w=out_w, out_h=out_h, preset=args.nvenc_preset, cq=args.nvenc_cq,
    )

    # ---- 每个新视频清空稳定器状态 ----
    stabilizer.reset()

    stage_times: Dict[str, float] = {}
    processed_frames = 0
    t_video0 = time.perf_counter()

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

            # ---------- Step 1: 双分辨率联合预处理（生成 DIBR 底图 + 稳定器 RGB） ----------
            # 注意：model_input 不再用于 VDA（VDA 内部自己处理预处理），保留该函数调用
            # 以复用 left_rgb_dibr 和 rgb_depth 的生成逻辑
            t0 = time.perf_counter()
            _, left_rgb_dibr, rgb_depth = prepare_inputs_dual_res_gpu(
                frame_bgr, device, (depth_h, depth_w), (dibr_h, dibr_w),
                mean, std, stage_times, args.profile_time,
            )
            _stage_add(stage_times, "preprocess_total", time.perf_counter() - t0)

            # ---------- Step 2: VDA 流式深度推理（边读边算，内部维护时序缓存） ----------
            t0 = time.perf_counter()
            depth_raw = infer_depth_raw_vda(
                model, frame_bgr, args.input_size, device, args.vda_fp32,
                depth_h, depth_w, stage_times, args.profile_time,
            )
            _stage_add(stage_times, "infer_depth_total", time.perf_counter() - t0)

            # ---------- Step 3: 时序稳定 (Plan A+B+C+D+E) ----------
            t0 = time.perf_counter()
            near_smooth = stabilizer.step(depth_raw, rgb_depth, stage_times, args.profile_time)
            _stage_add(stage_times, "stabilize_total", time.perf_counter() - t0)

            # ---------- Step 4: GPU 双线性重采样 near 到 DIBR 分辨率 (上 / 下采样均可) ----------
            t0 = time.perf_counter()
            near_dibr = F.interpolate(
                near_smooth[None, None, :, :],
                size=(dibr_h, dibr_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            _maybe_sync(device, args.profile_time)
            _stage_add(stage_times, "near_resample_gpu", time.perf_counter() - t0)

            # ---------- Step 5: near → disparity ----------
            t0 = time.perf_counter()
            disparity_dibr = near_dibr * max_disparity_dibr
            _maybe_sync(device, args.profile_time)
            _stage_add(stage_times, "near_to_disparity", time.perf_counter() - t0)

            # ---------- Step 6: int64 Z-buffer DIBR ----------
            t0 = time.perf_counter()
            right_rgb_dibr, hole_dibr = forward_warp_right_gpu(
                left_rgb_dibr, disparity_dibr, near_dibr, stage_times, args.profile_time,
            )
            _stage_add(stage_times, "gpu_warp_total", time.perf_counter() - t0)

            # ---------- Step 7: Fast inpaint ----------
            t0 = time.perf_counter()
            right_inpainted_dibr = fast_inpaint_gpu(
                right_rgb_dibr, hole_dibr, args.fast_kernel, args.fast_max_iter,
                stage_times, args.profile_time,
            )
            _stage_add(stage_times, "fast_inpaint_total", time.perf_counter() - t0)

            # ---------- Step 8: GPU → CPU ----------
            t0 = time.perf_counter()
            left_u8_dibr = (left_rgb_dibr.clamp(0, 1) * 255.0).byte().contiguous().cpu().numpy()
            right_u8_dibr = (
                right_inpainted_dibr.clamp(0, 1) * 255.0
            ).byte().contiguous().cpu().numpy()
            _stage_add(stage_times, "to_cpu_numpy", time.perf_counter() - t0)

            # ---------- Step 9: CPU 兜底拉回原分辨率 ----------
            t0 = time.perf_counter()
            if dibr_h == h_orig and dibr_w == w_orig:
                left_u8 = left_u8_dibr
                right_u8 = right_u8_dibr
            else:
                left_u8 = cv2.resize(left_u8_dibr, (w_orig, h_orig),
                                     interpolation=cv2.INTER_LINEAR)
                right_u8 = cv2.resize(right_u8_dibr, (w_orig, h_orig),
                                      interpolation=cv2.INTER_LINEAR)
            _stage_add(stage_times, "upsample_to_orig_cpu", time.perf_counter() - t0)

            # ---------- Step 10 & 11: 拼接 + 编码 ----------
            t0 = time.perf_counter()
            stereo = compose_stereo(left_u8, right_u8, args.layout, args.overlay_alpha)
            _stage_add(stage_times, "compose_stereo", time.perf_counter() - t0)

            t0 = time.perf_counter()
            writer.stdin.write(stereo.tobytes())
            _stage_add(stage_times, "write_to_ffmpeg", time.perf_counter() - t0)

            _stage_add(stage_times, "total_per_frame", time.perf_counter() - frame_t0)
            processed_frames += 1
            if processed_frames % 30 == 0:
                print(f"[mono2stereo] {Path(filename).name} rendered {processed_frames} frames")
    finally:
        reader.stop()
        cap.release()
        writer.stdin.close()
        ret = writer.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg 编码失败，退出码: {ret}")

    elapsed = time.perf_counter() - t_video0
    return stage_times, processed_frames, elapsed


# =============================================================================
# 性能输出
# =============================================================================

def print_profile(
        stage_times: Dict[str, float], processed_frames: int, elapsed: float,
) -> None:
    if processed_frames <= 0:
        return

    print("\n[mono2stereo] 各阶段用时统计:")
    print("=" * 70)

    top_level = [
        ("read_frame_queue", "从队列取帧 (后台预读)"),
        ("preprocess_total", "⚡ 输入预处理总耗时"),
        ("infer_depth_total", "⚡ 深度推理总耗时"),
        ("stabilize_total", "🌟 时序稳定总耗时 (Plan A+B+C+D+E)"),
        ("near_resample_gpu", "⚡ near 空间 GPU 双线性重采样到 DIBR 分辨率"),
        ("near_to_disparity", "near → disparity"),
        ("gpu_warp_total", "⚡ DIBR 图像扭曲总耗时 (int64)"),
        ("fast_inpaint_total", "⚡ 快速补洞总耗时"),
        ("to_cpu_numpy", "结果转回 CPU"),
        ("upsample_to_orig_cpu", "上采样回原分辨率 (CPU)"),
        ("compose_stereo", "立体帧拼接合成"),
        ("write_to_ffmpeg", "写入 ffmpeg 编码"),
    ]
    top_total = sum(stage_times.get(k, 0.0) for k, _ in top_level)
    print(f"📊 【顶层阶段汇总】 (总和: {top_total:.3f}s)")
    for key, name in top_level:
        if key in stage_times:
            t = stage_times[key]
            ratio = (t / top_total * 100.0) if top_total > 0 else 0.0
            print(f"  {key:30s} {t:8.3f}s ({ratio:5.1f}%)  |  {name}")

    print()
    print("🔍 【各阶段内部细分明细】")
    print("-" * 70)

    sub_groups: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("preprocess_total", [
            ("prep_h2d", "原图 BGR 传到 GPU"),
            ("prep_color_and_resize_gpu", "BGR→RGB + 双分辨率 Resize (全 GPU)"),
            ("prep_normalize", "Normalize 标准化 (GPU)"),
        ]),
        ("infer_depth_total", [
            ("model_inference", "⭐ Video-Depth-Anything 模型推理"),
        ]),
        ("stabilize_total", [
            ("stab_quantile_ema", "🅰 分位数 EMA + 归一化"),
            ("stab_near_calc", "near 转换 (1-d / d)"),
            ("stab_flow_align", "🅳 光流对齐 (Farneback + warp)"),
            ("stab_rgb_adaptive_ema", "🅱 RGB 引导自适应 EMA"),
            ("stab_median_window", "🅴 滑窗逐像素中值"),
        ]),
        ("gpu_warp_total", [
            ("warp_grid_gen", "生成网格坐标"),
            ("warp_index_prep", "准备线性索引"),
            ("warp_z_buffer", "无损 Z-buffer (int64 编码)"),
            ("warp_scatter_pixels", "像素散射"),
        ]),
    ]
    for parent_key, subs in sub_groups:
        if parent_key not in stage_times:
            continue
        pt = stage_times[parent_key]
        print(f"\n[{parent_key}] 内部拆解 (父阶段总用时: {pt:.3f}s):")
        for k, name in subs:
            if k in stage_times:
                t = stage_times[k]
                ratio = (t / pt * 100.0) if pt > 0 else 0.0
                print(f"  ├─ {k:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {name}")

    if "fast_inpaint_total" in stage_times:
        pt = stage_times["fast_inpaint_total"]
        print(f"\n[fast_inpaint_total] 内部拆解 (父阶段总用时: {pt:.3f}s):")
        for k in sorted([k for k in stage_times if k.startswith("inpaint_")]):
            t = stage_times[k]
            ratio = (t / pt * 100.0) if pt > 0 else 0.0
            if k == "inpaint_skip_no_holes":
                name = "跳过 (无空洞)"
            elif k == "inpaint_init_kernel":
                name = "初始化卷积核"
            elif k == "inpaint_final_fill_remaining":
                name = "剩余空洞兜底"
            elif k.startswith("inpaint_conv_"):
                iters = k.replace("inpaint_conv_", "").replace("iter", "")
                name = f"卷积迭代 ({iters} 次)"
            else:
                name = ""
            print(f"  ├─ {k:25s} {t:8.3f}s ({ratio:5.1f}%)  |  {name}")

    print()
    print("=" * 70)
    print(f"📈 处理帧数: {processed_frames}, "
          f"平均 FPS: {processed_frames / max(elapsed, 1e-6):.2f}")
    print("💡 提示: 'read_frame_queue' 极短说明预读队列已跑满，GPU 吃满了！")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    script_t0 = time.perf_counter()
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("该脚本要求 CUDA。")

    torch.backends.cudnn.benchmark = True

    # ---- VDA 模型（Video-Depth-Anything 流式版本） ----
    model = VideoDepthAnything(**get_model_config(args.encoder))
    # 自动选择 relative / metric checkpoint
    ckpt_prefix = "metric_video_depth_anything" if args.vda_metric else "video_depth_anything"
    ckpt = args.ckpt or f"./checkpoints/{ckpt_prefix}_{args.encoder}.pth"
    print(f"[mono2stereo] loading VDA checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    # VDA 默认 fp16 推理（内部 autocast），--vda-fp32 使用 fp32

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # ---- 输入文件 ----
    files = collect_video_files(args.video_path)
    if not files:
        raise FileNotFoundError(f"未找到可处理视频: {args.video_path}")

    is_single_file = len(files) == 1 and args.output != "./output.mp4"
    os.makedirs(args.outdir, exist_ok=True)

    # ---- 时序稳定器（跨视频复用，每视频开始 reset）----
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
        f"[mono2stereo] VDA 模型: encoder={args.encoder}, "
        f"metric={'ON' if args.vda_metric else 'OFF'}, "
        f"fp32={'ON' if args.vda_fp32 else 'OFF (fp16)'}"
    )
    print(
        f"[mono2stereo] 时序稳定器配置 (VDA 已有时序一致性，以下为补充平滑): "
        f"quantile_smooth={args.quantile_smooth} (A), "
        f"depth_smooth={args.depth_smooth} (B+C), "
        f"rgb_motion_sigma={args.rgb_motion_sigma} (B), "
        f"flow_align={'ON' if args.flow_align else 'OFF'}"
        f"{f' (h={args.flow_height})' if args.flow_align else ''} (D), "
        f"median_window={args.median_window} (E)"
    )

    warmup_state = {"done": False}
    all_processed_frames = 0

    for idx, filename in enumerate(files):
        print(f"\n[mono2stereo] Progress {idx + 1}/{len(files)}: {filename}")

        if is_single_file:
            out_path = args.output
        else:
            out_path = os.path.join(args.outdir, f"{Path(filename).stem}_3d.mp4")

        stage_times, processed_frames, elapsed = process_one_video(
            filename, out_path, args, device, model, stabilizer,
            MEAN, STD, warmup_state,
        )
        if processed_frames == 0:
            continue

        all_processed_frames += processed_frames

        avg_fps = processed_frames / max(elapsed, 1e-6)
        print(f"[mono2stereo] output={out_path}")
        print(f"[mono2stereo] processed_frames={processed_frames}, "
              f"total_time={elapsed:.3f}s, avg_fps={avg_fps:.3f}")

        if args.profile_time:
            print_profile(stage_times, processed_frames, elapsed)

    if args.profile_total:
        script_elapsed = time.perf_counter() - script_t0
        script_avg_fps = all_processed_frames / max(script_elapsed, 1e-6)
        print(f"\n[mono2stereo][total] frames={all_processed_frames}, "
              f"avg_fps={script_avg_fps:.3f}")


if __name__ == "__main__":
    main()
