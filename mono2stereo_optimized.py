# ==================== 【优化版：边缘引导深度平滑 + TV梯度空洞修补】 ====================
#   ✅ 新增1: 双边滤波/引导滤波的边缘感知深度平滑（消除晕轮）
#   ✅ 新增2: TV-L1 梯度驱动的空洞修补（更自然的边缘过渡）
#   ✅ 新增3: 边缘检测 + 梯度方向引导的扩散填充
# =============================================================================

import sys
sys.path.insert(0, '.')

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

try:
    from submodules.Video_Depth_Anything.video_depth_anything.video_depth_stream import VideoDepthAnything
    VIDEO_MODEL_AVAILABLE = True
except ImportError:
    VIDEO_MODEL_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D转3D优化版: 边缘引导平滑 + TV梯度修补")
    parser.add_argument("--video-path", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", type=str, default="./output.mp4", help="输出视频路径")
    parser.add_argument("--outdir", type=str, default="./vis_video_3d", help="输出目录")

    parser.add_argument("--input-size", type=int, default=512, help="深度模型输入高度")
    parser.add_argument("--dibr-size", type=int, default=0, help="DIBR渲染高度")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--video-model", action="store_true", help="使用视频模型")
    parser.add_argument("--metric", action="store_true", help="Metric深度模式")
    parser.add_argument("--ckpt", type=str, default=None, help="模型权重路径")
    parser.add_argument("--fp16", action="store_true", help="启用FP16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--queue-size", type=int, default=8)

    parser.add_argument("--max-disparity", type=float, default=16.0)
    parser.add_argument("--depth-mode", choices=["metric", "inverse"], default="inverse")
    parser.add_argument("--clip-low", type=float, default=0.01)
    parser.add_argument("--clip-high", type=float, default=0.99)

    # ========== 【新增：边缘引导平滑参数】 ==========
    parser.add_argument("--edge-aware-smooth", action="store_true", default=True,
                        help="启用边缘感知深度平滑（双边滤波）")
    parser.add_argument("--no-edge-aware-smooth", action="store_false", dest="edge_aware_smooth",
                        help="禁用边缘感知深度平滑")
    parser.add_argument("--smooth-sigma-spatial", type=float, default=5.0,
                        help="双边滤波空间标准差（越大平滑范围越大）")
    parser.add_argument("--smooth-sigma-color", type=float, default=0.1,
                        help="双边滤波颜色标准差（越小边缘保留越强）")
    parser.add_argument("--smooth-sigma-depth", type=float, default=0.05,
                        help="双边滤波深度标准差（越小深度不连续处越不平滑）")
    parser.add_argument("--guided-filter-radius", type=int, default=4,
                        help="引导滤波半径（替代双边滤波的更快选择）")
    parser.add_argument("--guided-filter-eps", type=float, default=1e-4,
                        help="引导滤波正则化参数")

    # ========== 【新增：TV梯度修补参数】 ==========
    parser.add_argument("--tv-inpaint", action="store_true", default=True,
                        help="启用TV-L1梯度驱动的空洞修补")
    parser.add_argument("--no-tv-inpaint", action="store_false", dest="tv_inpaint",
                        help="禁用TV-L1梯度修补，使用简单填充")
    parser.add_argument("--tv-lambda", type=float, default=0.1,
                        help="TV正则化强度（越大越平滑）")
    parser.add_argument("--tv-max-iter", type=int, default=50,
                        help="TV迭代次数")
    parser.add_argument("--tv-tau", type=float, default=0.125,
                        help="TV梯度下降步长")
    parser.add_argument("--edge-first-inpaint", action="store_true", default=True,
                        help="TV修补时先修复边缘")
    parser.add_argument("--no-edge-first-inpaint", action="store_false", dest="edge_first_inpaint",
                        help="TV修补时不区分边缘")

    # ========== 【新增：各向异性扩散修补（导师推荐的梯度方法）】 ==========
    parser.add_argument("--anisotropic-inpaint", action="store_true", default=False,
                        help="启用各向异性扩散修补（梯度方法，边缘保持好）")
    parser.add_argument("--no-anisotropic-inpaint", action="store_false", dest="anisotropic_inpaint",
                        help="禁用各向异性扩散，使用原算法或TV")
    parser.add_argument("--aniso-kappa", type=float, default=0.05,
                        help="各向异性扩散边缘敏感度，越小越保边，推荐0.02-0.1")
    parser.add_argument("--aniso-max-iter", type=int, default=100,
                        help="各向异性扩散迭代次数，推荐50-200")

    # 原有时序稳定参数
    parser.add_argument("--quantile-smooth", type=float, default=0.8)
    parser.add_argument("--depth-smooth", type=float, default=0.6)
    parser.add_argument("--rgb-motion-sigma", type=float, default=0.02)
    parser.add_argument("--flow-align", action="store_true", default=True)
    parser.add_argument("--flow-height", type=int, default=144)
    parser.add_argument("--median-window", type=int, default=3)

    parser.add_argument("--hole-dilate-left", type=int, default=2)
    parser.add_argument("--hole-dilate-right", type=int, default=1)

    parser.add_argument("--layout", choices=["sbs", "ou", "overlay", "anaglyph"], default="sbs")
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument("--anaglyph-mode", choices=["red-cyan", "color"], default="red-cyan")

    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    parser.add_argument("--video-encoder", type=str, default="h264_nvenc")
    parser.add_argument("--nvenc-gpu", type=int, default=0)
    parser.add_argument("--nvenc-preset", type=str, default="medium")
    parser.add_argument("--nvenc-cq", type=int, default=19)
    parser.add_argument("--profile-time", action="store_true")
    parser.add_argument("--profile-total", action="store_true")
    parser.add_argument("--profile-output", type=str, default=None)

    return parser.parse_args()


# ==================== 【优化1: 边缘感知深度平滑】 ====================

@torch.no_grad()
def bilateral_filter_depth_gpu(
    depth: torch.Tensor,
    guide_rgb: torch.Tensor,
    sigma_spatial: float,
    sigma_color: float,
    sigma_depth: float,
) -> torch.Tensor:
    """
    GPU加速的联合双边滤波（Joint Bilateral Filter）
    用RGB图像引导深度平滑，在边缘处停止扩散

    depth: [H, W] 深度图
    guide_rgb: [3, H, W] 引导RGB图像
    sigma_spatial: 空间高斯标准差
    sigma_color: 颜色域高斯标准差
    sigma_depth: 深度域高斯标准差
    """
    H, W = depth.shape
    device = depth.device

    # 核半径
    radius = int(3 * sigma_spatial)
    kernel_size = 2 * radius + 1

    # 生成空间权重（高斯）
    y, x = torch.meshgrid(
        torch.arange(kernel_size, device=device) - radius,
        torch.arange(kernel_size, device=device) - radius,
        indexing='ij'
    )
    spatial_weight = torch.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))  # [K, K]

    # 对每个像素，计算邻域加权和
    # 使用 unfold 实现高效滑动窗口
    depth_pad = F.pad(depth.unsqueeze(0).unsqueeze(0),
                      (radius, radius, radius, radius), mode='replicate')
    rgb_pad = F.pad(guide_rgb.unsqueeze(0),
                    (radius, radius, radius, radius), mode='replicate')

    # 展开成窗口
    depth_windows = F.unfold(depth_pad, kernel_size=kernel_size)  # [1, K², H*W]
    rgb_windows = F.unfold(rgb_pad, kernel_size=kernel_size)      # [1, 3*K², H*W]

    depth_windows = depth_windows.squeeze(0).T  # [H*W, K²]
    rgb_windows = rgb_windows.squeeze(0).T      # [H*W, 3*K²]

    # 中心像素值
    depth_center = depth.reshape(-1, 1)  # [H*W, 1]
    rgb_center = guide_rgb.reshape(3, -1).T  # [H*W, 3]

    # 计算颜色距离权重
    rgb_center_expand = rgb_center.unsqueeze(1).repeat(1, kernel_size**2, 1)  # [H*W, K², 3]
    rgb_windows_reshape = rgb_windows.reshape(-1, kernel_size**2, 3)  # [H*W, K², 3]
    color_dist_sq = ((rgb_windows_reshape - rgb_center_expand)**2).sum(dim=-1)  # [H*W, K²]
    color_weight = torch.exp(-color_dist_sq / (2 * sigma_color**2))

    # 计算深度距离权重
    depth_dist_sq = (depth_windows - depth_center)**2  # [H*W, K²]
    depth_weight = torch.exp(-depth_dist_sq / (2 * sigma_depth**2))

    # 总权重 = 空间权重 * 颜色权重 * 深度权重
    total_weight = spatial_weight.reshape(1, -1) * color_weight * depth_weight  # [H*W, K²]

    # 加权平均
    weight_sum = total_weight.sum(dim=1, keepdim=True)  # [H*W, 1]
    depth_smooth = (depth_windows * total_weight).sum(dim=1) / weight_sum.squeeze(1)

    return depth_smooth.reshape(H, W)


@torch.no_grad()
def guided_filter_depth_gpu(
    depth: torch.Tensor,
    guide_rgb: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """
    引导滤波（Guided Filter）- 比双边滤波快，边缘保持效果好
    适合实时视频处理
    """
    H, W = depth.shape
    device = depth.device

    # 转换到引导图像亮度通道
    guide = 0.299 * guide_rgb[0] + 0.587 * guide_rgb[1] + 0.114 * guide_rgb[2]

    # 盒滤波核
    kernel_size = 2 * radius + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size**2)

    # 局部均值
    mean_I = F.conv2d(guide.unsqueeze(0).unsqueeze(0), kernel, padding=radius)[0, 0]
    mean_p = F.conv2d(depth.unsqueeze(0).unsqueeze(0), kernel, padding=radius)[0, 0]

    # 局部协方差
    mean_II = F.conv2d((guide * guide).unsqueeze(0).unsqueeze(0), kernel, padding=radius)[0, 0]
    mean_Ip = F.conv2d((guide * depth).unsqueeze(0).unsqueeze(0), kernel, padding=radius)[0, 0]

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # 线性系数
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 均值滤波后的系数
    mean_a = F.conv2d(a.unsqueeze(0).unsqueeze(0), kernel, padding=radius)[0, 0]
    mean_b = F.conv2d(b.unsqueeze(0).unsqueeze(0), kernel, padding=radius)[0, 0]

    # 输出
    q = mean_a * guide + mean_b
    return q


# ==================== 【优化2: TV-L1 梯度驱动空洞修补】 ====================

@torch.no_grad()
def compute_gradient_map(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算图像梯度（Sobel算子）"""
    # Sobel核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=img.device, dtype=img.dtype) / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=img.device, dtype=img.dtype) / 4

    # 转灰度
    if img.dim() == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img

    # 计算梯度
    grad_x = F.conv2d(gray.unsqueeze(0).unsqueeze(0),
                       sobel_x.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
    grad_y = F.conv2d(gray.unsqueeze(0).unsqueeze(0),
                       sobel_y.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]

    return grad_x, grad_y


@torch.no_grad()
def anisotropic_inpaint_gpu(
    img: torch.Tensor,
    mask: torch.Tensor,
    max_iter: int,
    kappa: float,
    stage_times: Dict[str, float],
    profile_sync: bool,
) -> torch.Tensor:
    """
    导师推荐的【梯度方法】：各向异性扩散（Perona-Malik 方程）
    沿图像边缘切线方向扩散，垂直边缘方向不扩散，保持边缘锐利。

    核心公式：∂u/∂t = div( g(|∇u|) · ∇u )
    其中 g(|∇u|) = exp( -(|∇u|/k)² ) 是边缘停止函数

    img: [H, W, 3] RGB图像 (0-1)
    mask: [H, W] bool, True=空洞
    max_iter: 迭代次数
    kappa: 边缘敏感度阈值，越小越不跨边缘，越大越平滑
    """
    H, W, C = img.shape
    device = img.device

    t0 = time.perf_counter()

    # 初始化：先做 3x3 均值扩散填充边界
    result = img.clone()
    known = ~mask.clone()

    # 先快速扩散初始化边缘
    kernel = torch.ones(3, 3, device=device)
    for _ in range(3):
        if not torch.any(mask):
            break
        known_float = known.float().unsqueeze(-1)
        neighbor_sum = F.conv2d(
            (result * known_float).permute(2, 0, 1).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1),
            padding=1, groups=C
        )[0].permute(1, 2, 0)
        neighbor_count = F.conv2d(
            known_float.permute(2, 0, 1).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        )[0, 0]
        valid = neighbor_count > 0
        fillable = mask & valid
        if not torch.any(fillable):
            break
        result[fillable] = neighbor_sum[fillable] / neighbor_count[fillable].unsqueeze(-1)
        known[fillable] = True
        mask[fillable] = False

    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_aniso_init", time.perf_counter() - t0)

    # ========== 各向异性扩散主循环 ==========
    t0 = time.perf_counter()

    # 迭代步长（时间步长，保证稳定性）
    dt = 0.125

    for i in range(max_iter):
        if not torch.any(mask):
            break

        # --- 步骤1：对每个通道计算梯度（只在已知像素区域计算）---
        # 转为灰度计算梯度方向（三通道共享同一个结构张量）
        gray = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]

        # 中心差分计算梯度 [H, W]
        grad_x = torch.zeros_like(gray)
        grad_y = torch.zeros_like(gray)
        grad_x[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
        grad_y[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])

        # 梯度幅度
        grad_mag_sq = grad_x**2 + grad_y**2 + 1e-8

        # --- 步骤2：边缘停止函数 g(|∇u|) ---
        # 梯度大（边缘）：g → 0  不扩散
        # 梯度小（平坦）：g → 1  正常扩散
        g = torch.exp(-grad_mag_sq / (kappa**2))

        # --- 步骤3：四邻域扩散（Perona-Malik 式 1）---
        # 计算四个方向的传导率
        g_right = torch.minimum(g[:, :-1], g[:, 1:])  # x 方向
        g_left = g_right
        g_down = torch.minimum(g[:-1, :], g[1:, :])   # y 方向
        g_up = g_down

        # 对每个通道进行扩散更新
        for c in range(C):
            u = result[:, :, c]

            # 四方向差分
            diff_right = u[:, 1:] - u[:, :-1]
            diff_left = -diff_right
            diff_down = u[1:, :] - u[:-1, :]
            diff_up = -diff_down

            # 散度更新
            delta_u = torch.zeros_like(u)
            delta_u[:, :-1] = delta_u[:, :-1] + g_right * diff_right
            delta_u[:, 1:] = delta_u[:, 1:] + g_left * diff_left
            delta_u[:-1, :] = delta_u[:-1, :] + g_down * diff_down
            delta_u[1:, :] = delta_u[1:, :] + g_up * diff_up

            # 只更新空洞区域（注意 PyTorch 索引方式）
            u_out = u.clone()
            u_out[mask] = u[mask] + dt * delta_u[mask]
            result[:, :, c] = u_out

        # 更新已填充的空洞（只要扩散了就标记为已知，参与下一轮梯度计算）
        # 注意：mask 保持不变，只是梯度计算包含了已填充区域

    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, f"inpaint_aniso_iter{i+1}", time.perf_counter() - t0)

    return result


@torch.no_grad()
def tv_l1_inpaint_gpu(
    img: torch.Tensor,
    mask: torch.Tensor,
    tv_lambda: float,
    max_iter: int,
    tau: float,
    stage_times: Dict[str, float],
    profile_sync: bool,
) -> torch.Tensor:
    """
    TV-L1 变分空洞修补
    最小化能量: E(u) = ||∇u||_1 + λ||(u - u0) * (1 - m)||_1
    用梯度方向引导填充，保持边缘自然

    img: [H, W, 3] RGB图像 (0-1)
    mask: [H, W] bool, True=空洞
    tv_lambda: 数据项权重
    max_iter: 迭代次数
    tau: 梯度下降步长
    """
    H, W, C = img.shape
    device = img.device

    t0 = time.perf_counter()

    # 初始化：用已知像素填充边界
    result = img.clone()
    known = ~mask

    # 先做一次简单的边界扩散初始化
    kernel = torch.ones(3, 3, device=device)
    for _ in range(3):
        known_float = known.float().unsqueeze(-1)
        neighbor_sum = F.conv2d(
            (result * known_float).permute(2, 0, 1).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1),
            padding=1, groups=C
        )[0].permute(1, 2, 0)
        neighbor_count = F.conv2d(
            known_float.permute(2, 0, 1).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        )[0, 0]
        valid = neighbor_count > 0
        result[mask & valid] = neighbor_sum[mask & valid] / neighbor_count[mask & valid].unsqueeze(-1)
        known = known | (mask & valid)

    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_tv_init", time.perf_counter() - t0)

    # TV-L1 主迭代（Chambolle-Pock 算法）
    t0 = time.perf_counter()

    # 转换为浮点张量
    u = result.clone()
    u0 = img.clone()

    # 对偶变量 [H, W, 3, 2] - 每个通道独立的梯度
    p = torch.zeros(H, W, C, 2, device=device, dtype=u.dtype)

    # 数据项权重图（在空洞区域为0，边界区域过渡）
    data_weight = (1 - mask.float()).unsqueeze(-1)  # [H, W, 1]

    # TV迭代
    for i in range(max_iter):
        # 原始变量梯度
        grad_u_x = torch.roll(u, -1, 1) - u  # [H, W, 3]
        grad_u_y = torch.roll(u, -1, 0) - u
        grad_u = torch.stack([grad_u_x, grad_u_y], dim=-1)  # [H, W, 3, 2]

        # 对偶上升
        p = p + tau * grad_u
        norm_p = torch.clamp(torch.sqrt((p**2).sum(dim=-1, keepdim=True)), min=1.0)
        p = p / norm_p

        # 原始下降: div p
        div_p = (p[:, :, :, 0] - torch.roll(p[:, :, :, 0], 1, 1) +
                 p[:, :, :, 1] - torch.roll(p[:, :, :, 1], 1, 0))

        # 更新空洞区域
        data_term = tv_lambda * data_weight * (u - u0)
        update = div_p - data_term
        u[mask] = (u - tau * update)[mask]

    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_tv_iterations", time.perf_counter() - t0)

    return u


# ==================== 【优化3: EdgeConnect 风格两阶段修补】 ====================

@torch.no_grad()
def edge_aware_tv_inpaint_gpu(
    img: torch.Tensor,
    mask: torch.Tensor,
    near: Optional[torch.Tensor],
    tv_lambda: float,
    max_iter: int,
    tau: float,
    stage_times: Dict[str, float],
    profile_sync: bool,
) -> torch.Tensor:
    """
    EdgeConnect风格: 先用快速扩散填充，再用TV平滑
    简化版：移除复杂的梯度外插，直接用均值扩散 + TV
    """
    device = img.device
    H, W, C = img.shape

    t0 = time.perf_counter()

    # ========== 阶段1: 快速扩散初始化 ==========
    result = img.clone()
    known = ~mask

    # 3x3 均值扩散，迭代5次填充边缘
    kernel = torch.ones(3, 3, device=device)
    for _ in range(5):
        if not torch.any(mask):
            break
        known_float = known.float().unsqueeze(-1)
        neighbor_sum = F.conv2d(
            (result * known_float).permute(2, 0, 1).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1),
            padding=1, groups=C
        )[0].permute(1, 2, 0)
        neighbor_count = F.conv2d(
            known_float.permute(2, 0, 1).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        )[0, 0]
        valid = neighbor_count > 0
        fillable = mask & valid
        if not torch.any(fillable):
            break
        result[fillable] = neighbor_sum[fillable] / neighbor_count[fillable].unsqueeze(-1)
        known[fillable] = True
        mask[fillable] = False

    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_edge_first", time.perf_counter() - t0)

    # ========== 阶段2: TV填充内部区域 ==========
    if torch.any(mask):
        t0 = time.perf_counter()
        result = tv_l1_inpaint_gpu(
            result, mask, tv_lambda, max_iter, tau,
            stage_times, profile_sync
        )
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "inpaint_tv_second_stage", time.perf_counter() - t0)

    return result


# ==================== 原代码工具函数 ====================

def get_model_config(encoder: str) -> Dict:
    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    return configs[encoder]


def _stage_add(stage_times: Dict[str, float], key: str, dt: float) -> None:
    stage_times[key] = stage_times.get(key, 0.0) + dt


def _maybe_sync(device: torch.device, do_sync: bool) -> None:
    if do_sync and device.type == "cuda":
        torch.cuda.synchronize()


def detect_hole_edges(hole_mask: torch.Tensor) -> torch.Tensor:
    is_hole = hole_mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3), device=hole_mask.device, dtype=is_hole.dtype)
    neighbor_count = F.conv2d(1 - is_hole, kernel, padding=1)
    edge_mask = hole_mask & (neighbor_count[0, 0] > 0.01)
    return edge_mask


# ==================== 时序稳定器（集成边缘平滑） ====================

class TemporalDepthStabilizer:
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
            # 新增参数
            edge_aware_enabled: bool,
            sigma_spatial: float,
            sigma_color: float,
            sigma_depth: float,
            guided_radius: int,
            guided_eps: float,
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

        # Plan A
        self._low_ema: Optional[torch.Tensor] = None
        self._high_ema: Optional[torch.Tensor] = None

        # Plan B/D state
        self._prev_near: Optional[torch.Tensor] = None
        self._prev_rgb: Optional[torch.Tensor] = None

        # 新增：边缘感知平滑
        self.edge_aware_enabled = edge_aware_enabled
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.sigma_depth = sigma_depth
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    def reset(self) -> None:
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
        device = depth_raw.device

        # Plan A: 分位数 EMA
        t0 = time.perf_counter()
        flat = depth_raw.reshape(-1)
        sample_size = 16384
        idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
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

        # near 空间转换
        t0 = time.perf_counter()
        near_smooth = depth_norm if self.depth_mode == "inverse" else (1.0 - depth_norm)
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "stab_near_calc", time.perf_counter() - t0)

        # Plan D: 光流对齐
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

        # Plan B + C: RGB 引导的逐像素 EMA
        if self.depth_smooth > 0.0 and prev_near_aligned is not None:
            t0 = time.perf_counter()
            if self.rgb_motion_sigma > 0.0 and prev_rgb_aligned is not None:
                diff_sq = (rgb_depth - prev_rgb_aligned).pow(2).sum(dim=0)
                motion = 1.0 - torch.exp(-diff_sq / (self.rgb_motion_sigma ** 2))
                a_curr = 1.0 - self.depth_smooth * (1.0 - motion)
            else:
                a_curr = 1.0 - self.depth_smooth
            near_smooth = a_curr * near_smooth + (1.0 - a_curr) * prev_near_aligned
            _maybe_sync(device, profile_sync)
            _stage_add(stage_times, "stab_rgb_adaptive_ema", time.perf_counter() - t0)

        # ========== 【新增：边缘感知深度平滑】 ==========
        if self.edge_aware_enabled:
            t0 = time.perf_counter()
            # 用引导滤波（更快）或双边滤波（更好）
            if self.guided_radius > 0:
                near_smooth = guided_filter_depth_gpu(
                    near_smooth, rgb_depth,
                    self.guided_radius, self.guided_eps
                )
            else:
                near_smooth = bilateral_filter_depth_gpu(
                    near_smooth, rgb_depth,
                    self.sigma_spatial, self.sigma_color, self.sigma_depth
                )
            _maybe_sync(device, profile_sync)
            _stage_add(stage_times, "stab_edge_aware_smooth", time.perf_counter() - t0)

        # 更新状态
        self._prev_near = near_smooth.detach().clone()
        self._prev_rgb = rgb_depth.detach().clone()

        # Plan E: 滑窗中值
        if self.median_window > 1:
            t0 = time.perf_counter()
            self._near_history.append(near_smooth.detach())
            if len(self._near_history) >= 2:
                stack = torch.stack(list(self._near_history), dim=0)
                near_smooth = stack.median(dim=0).values
            _maybe_sync(device, profile_sync)
            _stage_add(stage_times, "stab_median_window", time.perf_counter() - t0)

        return near_smooth


# ==================== 光流相关（原代码保留） ====================

class OpticalFlowEstimator:
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
        device = rgb_curr.device
        H_in, W_in = rgb_curr.shape[-2:]
        flow_h = self.flow_height
        flow_w = max(2, int(round(W_in * flow_h / H_in)))

        gray = (0.299 * rgb_curr[0] + 0.587 * rgb_curr[1] + 0.114 * rgb_curr[2])
        gray_small = F.interpolate(
            gray[None, None], size=(flow_h, flow_w),
            mode="bilinear", align_corners=False,
        )[0, 0]
        gray_cpu = (gray_small.clamp(0, 1) * 255.0).byte().cpu().numpy()

        if self._prev_gray_cpu is None:
            self._prev_gray_cpu = gray_cpu
            return None

        flow_small = cv2.calcOpticalFlowFarneback(
            gray_cpu, self._prev_gray_cpu, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        self._prev_gray_cpu = gray_cpu

        flow_t = torch.from_numpy(flow_small).to(device).permute(2, 0, 1)
        H_t, W_t = target_size_hw
        flow_up = F.interpolate(
            flow_t[None], size=(H_t, W_t), mode="bilinear", align_corners=False,
        )[0]
        flow_up[0] *= W_t / float(flow_w)
        flow_up[1] *= H_t / float(flow_h)
        return flow_up


@torch.no_grad()
def backward_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    H, W = flow.shape[-2:]
    device = flow.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=flow.dtype),
        torch.arange(W, device=device, dtype=flow.dtype),
        indexing="ij",
    )
    x_sample = xx + flow[0]
    y_sample = yy + flow[1]
    x_norm = 2.0 * (x_sample + 0.5) / W - 1.0
    y_norm = 2.0 * (y_sample + 0.5) / H - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)[None]

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


# ==================== DIBR（原代码保留） ====================

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
def dilate_hole_right(hole: torch.Tensor, dilate_px: int) -> torch.Tensor:
    if dilate_px <= 0:
        return hole
    out = hole.clone()
    for shift in range(1, dilate_px + 1):
        out = out | torch.roll(hole, shifts=shift, dims=1)
    out[:, :dilate_px] = hole[:, :dilate_px]
    return out


# ==================== 【原算法：边缘敏感 inpaint（保留作为默认）】 ====================

@functools.lru_cache(maxsize=8)
def _get_inpaint_kernels(kernel_size: int, device: str, dtype: str):
    """缓存 inpaint kernel，避免每帧重建"""
    device = torch.device(device)
    dtype = getattr(torch, dtype)
    pad = kernel_size // 2
    kernel1 = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)
    return kernel1, kernel3


@torch.no_grad()
def fill_edge_with_nearest_bg(
    img: torch.Tensor, hole: torch.Tensor, edge_mask: torch.Tensor,
    near: Optional[torch.Tensor], bg_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """直接用空洞边缘周围的背景像素填补边缘区域"""
    result = img.clone()
    filled_mask = torch.zeros_like(hole)
    device = hole.device

    if near is not None:
        bg_mask = ~hole & (near < bg_threshold)
    else:
        bg_mask = ~hole

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


@torch.no_grad()
def fast_inpaint_gpu(
    image: torch.Tensor, hole_mask: torch.Tensor, kernel_size: int, max_iter: int,
    stage_times: Dict[str, float], profile_sync: bool,
    near: Optional[torch.Tensor] = None, bg_threshold: float = 0.3,
    edge_kernel_size: int = 5, non_edge_kernel_size: int = 11, edge_fill_mode: int = 0,
) -> torch.Tensor:
    """
    边缘敏感的空洞填补（原版算法，效果有保证）：
    - edge_fill_mode=0: 边缘用小核，非边缘用大核
    - edge_fill_mode=1: 边缘直接用周围背景像素填补，非边缘用大核
    - edge_fill_mode=2: 混合模式，边缘先用背景像素填补，再统一处理
    """
    if edge_kernel_size % 2 == 0:
        raise ValueError("edge-kernel必须是奇数")
    if non_edge_kernel_size % 2 == 0:
        raise ValueError("non-edge-kernel必须是奇数")

    t0 = time.perf_counter()
    img = image.clone()
    hole = hole_mask.clone()
    if not torch.any(hole):
        _stage_add(stage_times, "inpaint_skip_no_holes", time.perf_counter() - t0)
        return img

    device = img.device

    # ========== 步骤1：检测空洞边缘 ==========
    edge_mask = detect_hole_edges(hole)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_detect_edges", time.perf_counter() - t0)

    # ========== 步骤2（模式1/2）：先直接填补边缘区域 ==========
    if edge_fill_mode in (1, 2):
        t0 = time.perf_counter()
        img, edge_filled = fill_edge_with_nearest_bg(img, hole, edge_mask, near, bg_threshold)
        hole[edge_filled] = False
        _maybe_sync(device, profile_sync)
        _stage_add(stage_times, "inpaint_edge_direct_fill", time.perf_counter() - t0)

    # ========== 步骤3：获取两个卷积核 ==========
    edge_pad = edge_kernel_size // 2
    non_edge_pad = non_edge_kernel_size // 2
    edge_kernel1, edge_kernel3 = _get_inpaint_kernels(
        edge_kernel_size, str(device), str(img.dtype).split('.')[-1]
    )
    non_edge_kernel1, non_edge_kernel3 = _get_inpaint_kernels(
        non_edge_kernel_size, str(device), str(img.dtype).split('.')[-1]
    )

    # ========== 步骤4：构建背景权重 ==========
    if near is not None:
        is_bg = (near < bg_threshold).to(img.dtype)
        bg_w = is_bg.unsqueeze(0).unsqueeze(0)
    else:
        bg_w = torch.ones((1, 1, *hole.shape), device=device, dtype=img.dtype)

    _maybe_sync(device, profile_sync)

    # ========== 步骤5：迭代填补 ==========
    t0 = time.perf_counter()
    iter_count = 0
    for _ in range(max_iter):
        iter_count += 1
        if not torch.any(hole):
            break

        filled_this_iter = torch.zeros_like(hole)
        known = (~hole).float().unsqueeze(0).unsqueeze(0)
        w = known * bg_w

        # 分开处理边缘和非边缘区域
        # --- 边缘区域：用小卷积核 ---
        current_edge_mask = detect_hole_edges(hole) & hole

        if torch.any(current_edge_mask):
            edge_count = F.conv2d(w, edge_kernel1, padding=edge_pad)
            edge_fillable = current_edge_mask & (edge_count[0, 0] > 0.01)

            if torch.any(edge_fillable):
                img_nchw = img.permute(2, 0, 1).unsqueeze(0)
                edge_rgb_sum = F.conv2d(img_nchw * w, edge_kernel3, padding=edge_pad, groups=3)
                edge_avg = edge_rgb_sum / edge_count.clamp_min(1e-6)
                edge_avg_hwc = edge_avg[0].permute(1, 2, 0)

                img[edge_fillable] = edge_avg_hwc[edge_fillable]
                hole[edge_fillable] = False
                filled_this_iter |= edge_fillable

        # --- 非边缘区域：用大卷积核 ---
        if torch.any(hole):
            non_edge_mask = hole & (~current_edge_mask)

            if torch.any(non_edge_mask):
                non_edge_count = F.conv2d(w, non_edge_kernel1, padding=non_edge_pad)
                non_edge_fillable = non_edge_mask & (non_edge_count[0, 0] > 0.01)

                if torch.any(non_edge_fillable):
                    img_nchw = img.permute(2, 0, 1).unsqueeze(0)
                    non_edge_rgb_sum = F.conv2d(img_nchw * w, non_edge_kernel3, padding=non_edge_pad, groups=3)
                    non_edge_avg = non_edge_rgb_sum / non_edge_count.clamp_min(1e-6)
                    non_edge_avg_hwc = non_edge_avg[0].permute(1, 2, 0)

                    img[non_edge_fillable] = non_edge_avg_hwc[non_edge_fillable]
                    hole[non_edge_fillable] = False
                    filled_this_iter |= non_edge_fillable

        # 如果这轮没填补任何像素，说明周围没有可用背景了，退出
        if not torch.any(filled_this_iter):
            break

    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, f"inpaint_edge_aware_{iter_count}iter", time.perf_counter() - t0)

    # ========== 步骤6：兜底填充（如果还有剩下的洞） ==========
    t0 = time.perf_counter()
    if torch.any(hole):
        for _ in range(max_iter // 2):
            if not torch.any(hole):
                break

            fallback_known = (~hole).float().unsqueeze(0).unsqueeze(0)
            count = F.conv2d(fallback_known, non_edge_kernel1, padding=non_edge_pad)
            fillable = hole & (count[0, 0] > 0.01)

            if not torch.any(fillable):
                break

            img_nchw = img.permute(2, 0, 1).unsqueeze(0)
            rgb_sum = F.conv2d(img_nchw * fallback_known, non_edge_kernel3, padding=non_edge_pad, groups=3)
            avg = rgb_sum / count.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            img[fillable] = avg_hwc[fillable]
            hole[fillable] = False

    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "inpaint_final_fallback", time.perf_counter() - t0)

    # 最差情况：还有洞，用全图平均
    if torch.any(hole):
        known_pixels = image[~hole_mask]
        fallback = known_pixels.mean(dim=0) if known_pixels.numel() > 0 else torch.zeros(3, device=device, dtype=img.dtype)
        img[hole] = fallback

    return img


# ==================== 统一的空洞修补入口 ====================

@torch.no_grad()
def inpaint_combined(
    img: torch.Tensor,
    hole: torch.Tensor,
    near: Optional[torch.Tensor],
    use_anisotropic: bool,      # 新增：各向异性扩散（优先级最高）
    aniso_kappa: float,         # 新增：边缘敏感度
    aniso_max_iter: int,        # 新增：迭代次数
    use_tv: bool,
    tv_lambda: float,
    tv_max_iter: int,
    tv_tau: float,
    edge_first: bool,
    stage_times: Dict[str, float],
    profile_sync: bool,
) -> torch.Tensor:
    """统一的修补入口（优先级从高到低）：
    1. use_anisotropic=True: 各向异性扩散（导师推荐的梯度方法，边缘保持好）
    2. use_tv=True:  TV-L1 梯度修补（实验性）
    3. 默认: 原算法的边缘敏感卷积 inpaint（效果有保证）
    """
    if not torch.any(hole):
        return img

    if use_anisotropic:
        # 各向异性扩散（导师推荐的梯度方法）
        return anisotropic_inpaint_gpu(
            img, hole, aniso_max_iter, aniso_kappa,
            stage_times, profile_sync
        )
    elif use_tv:
        # TV 修补（实验性）
        if edge_first:
            return edge_aware_tv_inpaint_gpu(
                img, hole, near, tv_lambda, tv_max_iter, tv_tau,
                stage_times, profile_sync
            )
        else:
            return tv_l1_inpaint_gpu(
                img, hole, tv_lambda, tv_max_iter, tv_tau,
                stage_times, profile_sync
            )
    else:
        # 原算法：边缘敏感卷积 inpaint（效果有保证，默认）
        return fast_inpaint_gpu(
            img, hole_mask=hole,
            kernel_size=11, max_iter=64,  # 原算法默认参数
            stage_times=stage_times, profile_sync=profile_sync,
            near=near, bg_threshold=0.3,
            edge_kernel_size=5, non_edge_kernel_size=11, edge_fill_mode=0
        )


# ==================== 剩余的主函数（精简，保持原逻辑） ====================

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


def compute_aspect_preserved_size(orig_h: int, orig_w: int, long_edge: int) -> Tuple[int, int]:
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
        result = np.zeros_like(left_u8)
        result[..., 0] = left_u8[..., 0]
        result[..., 1] = right_u8[..., 1]
        result[..., 2] = right_u8[..., 2]
        return result
    a = float(np.clip(overlay_alpha, 0.0, 1.0))
    mixed = (1.0 - a) * left_u8.astype(np.float32) + a * right_u8.astype(np.float32)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def create_nvenc_writer(
    ffmpeg_bin: str, input_video: str, output_video: str,
    fps: float, out_w: int, out_h: int, encoder: str, nvenc_gpu: int, preset: str, cq: int,
) -> subprocess.Popen:
    pix_fmt = "yuv420p" if (out_w % 2 == 0 and out_h % 2 == 0) else "yuv444p"

    def build_cmd(enc: str) -> list:
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-", "-i", input_video,
            "-map", "0:v:0", "-map", "1:a?",
        ]
        if enc == "h264_nvenc":
            cmd.extend([
                "-c:v", "h264_nvenc", "-gpu", str(nvenc_gpu),
                "-preset", preset, "-rc", "vbr", "-cq", str(cq),
            ])
        elif enc == "libx264":
            cmd.extend([
                "-c:v", "libx264", "-preset", preset if preset in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"] else "medium",
                "-crf", str(cq),
            ])
        cmd.extend([
            "-pix_fmt", pix_fmt, "-c:a", "aac", "-movflags", "+faststart",
            "-shortest", output_video,
        ])
        return cmd

    actual_encoder = encoder
    cmd = build_cmd(actual_encoder)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


@torch.no_grad()
def prepare_inputs_dual_res_gpu(
    frame_bgr: np.ndarray, device: torch.device,
    depth_size_hw: Tuple[int, int], dibr_size_hw: Tuple[int, int],
    mean: Tuple[float, float, float], std: Tuple[float, float, float],
    stage_times: Dict[str, float], profile_sync: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t0 = time.perf_counter()
    bgr_hwc = torch.from_numpy(frame_bgr).to(
        device=device, dtype=torch.float32, non_blocking=True,
    )
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_h2d", time.perf_counter() - t0)

    t0 = time.perf_counter()
    rgb_nchw = bgr_hwc.flip(-1).permute(2, 0, 1).unsqueeze(0) / 255.0
    rgb_depth_nchw = F.interpolate(rgb_nchw, depth_size_hw, mode="bilinear", align_corners=False)
    if dibr_size_hw == (rgb_nchw.shape[2], rgb_nchw.shape[3]):
        rgb_dibr_nchw = rgb_nchw
    else:
        rgb_dibr_nchw = F.interpolate(rgb_nchw, dibr_size_hw, mode="bilinear", align_corners=False)
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_color_and_resize_gpu", time.perf_counter() - t0)

    t0 = time.perf_counter()
    rgb_depth_chw = rgb_depth_nchw[0].contiguous()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    model_input = (rgb_depth_nchw - mean_t) / std_t
    left_rgb_dibr_hwc = rgb_dibr_nchw[0].permute(1, 2, 0).contiguous()
    _maybe_sync(device, profile_sync)
    _stage_add(stage_times, "prep_normalize", time.perf_counter() - t0)

    return model_input, left_rgb_dibr_hwc, rgb_depth_chw


class FrameReaderThread(threading.Thread):
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
    depth = pred[0].float()
    return depth


def main() -> None:
    script_t0 = time.perf_counter()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("该脚本要求CUDA")

    torch.backends.cudnn.benchmark = True

    if args.video_model:
        if not VIDEO_MODEL_AVAILABLE:
            raise ImportError("Video-Depth-Anything 不可用")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        model = VideoDepthAnything(**model_configs[args.encoder])
        checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'
        ckpt = args.ckpt or f"checkpoints/{checkpoint_name}_{args.encoder}.pth"
        model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        model = model.to(device).eval()
        print(f"[mono2stereo] 使用 Video-Depth-Anything")
    else:
        model = DepthAnythingV2(**get_model_config(args.encoder))
        ckpt = args.ckpt or f"checkpoints/depth_anything_v2_{args.encoder}.pth"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        if args.fp16:
            model = model.half()
        print(f"[mono2stereo] 使用单帧模型 + PLAN A-E 时序稳定")

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    stabilizer = None
    if not args.video_model:
        stabilizer = TemporalDepthStabilizer(
            depth_mode=args.depth_mode,
            clip_low=args.clip_low, clip_high=args.clip_high,
            depth_smooth=args.depth_smooth, quantile_smooth=args.quantile_smooth,
            rgb_motion_sigma=args.rgb_motion_sigma,
            flow_align_enabled=args.flow_align, flow_height=args.flow_height,
            median_window=args.median_window,
            # 新增参数
            edge_aware_enabled=args.edge_aware_smooth,
            sigma_spatial=args.smooth_sigma_spatial,
            sigma_color=args.smooth_sigma_color,
            sigma_depth=args.smooth_sigma_depth,
            guided_radius=args.guided_filter_radius,
            guided_eps=args.guided_filter_eps,
        )
        print(f"[mono2stereo] 🌟 时序稳定器: quantile={args.quantile_smooth}, depth_smooth={args.depth_smooth}")
        if args.edge_aware_smooth:
            if args.guided_filter_radius > 0:
                print(f"[mono2stereo]  ✅ 边缘感知平滑: 引导滤波 r={args.guided_filter_radius}")
            else:
                print(f"[mono2stereo]  ✅ 边缘感知平滑: 双边滤波 spatial={args.smooth_sigma_spatial}")

    # 空洞修补配置
    if args.tv_inpaint:
        print(f"[mono2stereo] 🎨 TV-L1梯度修补: lambda={args.tv_lambda}, iter={args.tv_max_iter}")
        if args.edge_first_inpaint:
            print(f"[mono2stereo]  ✅ 边缘优先模式 (EdgeConnect风格)")

    files = collect_video_files(args.video_path)
    if not files:
        raise FileNotFoundError(f"未找到视频: {args.video_path}")

    is_single_file = len(files) == 1 and args.output != "./output.mp4"
    os.makedirs(args.outdir, exist_ok=True)
    all_processed_frames = 0

    for idx, filename in enumerate(files):
        print(f"\n[mono2stereo] 处理 {idx + 1}/{len(files)}: {filename}")
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print(f"[mono2stereo] 无法打开，跳过")
            continue

        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_orig = float(cap.get(cv2.CAP_PROP_FPS))
        fps_orig = fps_orig if fps_orig > 0 else 30.0

        # 深度推理尺寸
        long_edge = args.input_size
        scale = long_edge / max(h_orig, w_orig)
        depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
        depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
        if w_orig >= h_orig:
            depth_w = long_edge
        else:
            depth_h = long_edge

        # DIBR 尺寸
        if args.dibr_size == -1:
            dibr_h, dibr_w = h_orig, w_orig
        elif args.dibr_size == 0:
            dibr_h, dibr_w = depth_h, depth_w
        else:
            dibr_h = args.dibr_size
            dibr_w = int(round(dibr_h * (w_orig / h_orig) / 2.0)) * 2

        max_disparity_dibr = args.max_disparity * dibr_w / w_orig
        print(f"[mono2stereo] 原始: {w_orig}×{h_orig}, 深度: {depth_w}×{depth_h}, DIBR: {dibr_w}×{dibr_h}")

        if args.warmup_iters > 0 and not args.video_model:
            print("[mono2stereo] CUDA Warmup...")
            warmup_dtype = torch.float16 if args.fp16 else torch.float32
            dummy = torch.randn(1, 3, depth_h, depth_w, device=device, dtype=warmup_dtype)
            for _ in range(args.warmup_iters):
                _ = model(dummy)
            torch.cuda.synchronize()

        # 输出尺寸
        use_layout = args.layout
        if args.video_encoder == "h264_nvenc":
            sbs_w = w_orig * 2
            if use_layout == "sbs" and sbs_w > 4096:
                print(f"[mono2stereo] sbs宽度超限，切换到ou")
                use_layout = "ou"

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
            raise RuntimeError("ffmpeg初始化失败")

        stage_times: Dict[str, float] = {}
        processed_frames = 0

        if stabilizer is not None:
            stabilizer.reset()

        reader = FrameReaderThread(cap, queue_size=args.queue_size)
        reader.start()

        try:
            while True:
                ok, frame_bgr = reader.get_frame()
                if not ok:
                    break

                if args.video_model:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                        depth_raw_np = model.infer_video_depth_one(
                            frame_rgb, input_size=args.input_size, device=device, fp32=(not args.fp16)
                        )
                    depth_raw = torch.from_numpy(depth_raw_np).to(device=device, dtype=torch.float32)
                    left_rgb_dibr = torch.from_numpy(frame_bgr).to(device=device, dtype=torch.float32)
                    left_rgb_dibr = left_rgb_dibr.flip(-1) / 255.0
                    near_smooth = depth_raw
                else:
                    model_input, left_rgb_dibr, rgb_depth = prepare_inputs_dual_res_gpu(
                        frame_bgr, device, (depth_h, depth_w), (dibr_h, dibr_w),
                        MEAN, STD, stage_times, args.profile_time,
                    )
                    if args.fp16:
                        model_input = model_input.half()

                    depth_raw = infer_depth_lowres(
                        model, model_input, args.fp16, stage_times, args.profile_time
                    )

                    # 时序稳定（含边缘感知平滑）
                    t0 = time.perf_counter()
                    near_smooth = stabilizer.step(depth_raw, rgb_depth, stage_times, args.profile_time)
                    _stage_add(stage_times, "stabilize_total", time.perf_counter() - t0)

                # near -> disparity
                t0 = time.perf_counter()
                if args.video_model:
                    if args.depth_mode == "inverse":
                        near_v = depth_raw
                    else:
                        near_v = 1.0 - depth_raw
                    disparity = near_v * args.max_disparity
                    near_for_inpaint = near_v
                else:
                    if (dibr_h == depth_h and dibr_w == depth_w):
                        near_dibr = near_smooth
                    else:
                        near_dibr = F.interpolate(
                            near_smooth[None, None, :, :], size=(dibr_h, dibr_w),
                            mode="bilinear", align_corners=False,
                        )[0, 0]
                    disparity = near_dibr * max_disparity_dibr
                    near_for_inpaint = near_dibr
                _stage_add(stage_times, "near_to_disparity", time.perf_counter() - t0)

                # DIBR
                t0 = time.perf_counter()
                right_rgb_dibr, hole_dibr = forward_warp_right_gpu(
                    left_rgb_dibr, disparity, near_for_inpaint,
                    stage_times, args.profile_time
                )
                _stage_add(stage_times, "gpu_warp_total", time.perf_counter() - t0)

                # 空洞膨胀
                if args.hole_dilate_left > 0:
                    hole_dilated = dilate_hole_left(hole_dibr, args.hole_dilate_left)
                else:
                    hole_dilated = hole_dibr

                if args.hole_dilate_right > 0:
                    hole_dilated = dilate_hole_right(hole_dilated, args.hole_dilate_right)

                # ========== 【新增：空洞修补（优先级：各向异性 > TV > 原算法）】 ==========
                t0 = time.perf_counter()
                right_inpainted_dibr = inpaint_combined(
                    right_rgb_dibr, hole=hole_dilated, near=near_for_inpaint,
                    use_anisotropic=args.anisotropic_inpaint,      # 导师推荐的梯度方法
                    aniso_kappa=args.aniso_kappa,
                    aniso_max_iter=args.aniso_max_iter,
                    use_tv=args.tv_inpaint,
                    tv_lambda=args.tv_lambda,
                    tv_max_iter=args.tv_max_iter,
                    tv_tau=args.tv_tau,
                    edge_first=args.edge_first_inpaint,
                    stage_times=stage_times,
                    profile_sync=args.profile_time,
                )
                _stage_add(stage_times, "inpaint_total", time.perf_counter() - t0)

                # D2H + 上采样
                t0 = time.perf_counter()
                right_u8_dibr = (right_inpainted_dibr.clamp(0, 1) * 255.0).byte().contiguous().cpu().numpy()
                _stage_add(stage_times, "to_cpu_numpy", time.perf_counter() - t0)

                t0 = time.perf_counter()
                left_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if args.video_model or (dibr_h == h_orig and dibr_w == w_orig):
                    right_u8 = right_u8_dibr
                else:
                    right_u8 = cv2.resize(right_u8_dibr, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                _stage_add(stage_times, "upsample_to_orig_cpu", time.perf_counter() - t0)

                t0 = time.perf_counter()
                stereo = compose_stereo(left_u8, right_u8, use_layout, args.overlay_alpha)
                _stage_add(stage_times, "compose_stereo", time.perf_counter() - t0)

                t0 = time.perf_counter()
                writer.stdin.write(stereo.tobytes())
                _stage_add(stage_times, "write_to_ffmpeg", time.perf_counter() - t0)

                processed_frames += 1
                all_processed_frames += 1
                if processed_frames % 30 == 0:
                    print(f"[mono2stereo] 已渲染 {processed_frames} 帧")
        finally:
            reader.stop()
            cap.release()
            writer.stdin.close()
            stderr_output = writer.stderr.read().decode('utf-8', errors='replace')
            ret = writer.wait()
            if ret != 0:
                if stderr_output:
                    print(f"\n[mono2stereo] ❌ ffmpeg错误:\n{stderr_output}")
                raise RuntimeError(f"ffmpeg编码失败，退出码: {ret}")

        print(f"[mono2stereo] 输出: {out_path}, 帧数: {processed_frames}")

        if args.profile_time and processed_frames > 0:
            print(f"\n[mono2stereo] 各阶段耗时统计:")
            for key in sorted(stage_times.keys()):
                t = stage_times[key]
                t_avg_ms = t / processed_frames * 1000
                print(f"  {key:30s} {t:8.3f}s | {t_avg_ms:7.1f}ms/帧")

    if args.profile_total:
        script_elapsed = time.perf_counter() - script_t0
        script_avg_fps = all_processed_frames / max(script_elapsed, 1e-6)
        print(f"\n[mono2stereo][总] 帧数={all_processed_frames}, 平均FPS={script_avg_fps:.3f}")


if __name__ == "__main__":
    main()
