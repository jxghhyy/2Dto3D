#!/usr/bin/env python3
"""
对比 InfiniDepth 和 DepthAnythingV2 的深度输出
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules/InfiniDepth"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules/depth/dav2"))

import torch
import cv2
import numpy as np
from pathlib import Path

from InfiniDepth.utils.model_utils import build_model
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS
from depth_anything_v2.dpt import DepthAnythingV2


def test_infinidepth(img_path, device):
    """测试 InfiniDepth 深度"""
    print(f"\n{'='*60}")
    print(f"[InfiniDepth] 测试: {Path(img_path).name}")
    print(f"{'='*60}")

    # 加载模型
    model = build_model("InfiniDepth", model_path="checkpoints/infinidepth.ckpt")
    model = model.to(device).eval()

    # 加载图片
    img_bgr = cv2.imread(img_path)
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"[InfiniDepth] 原始尺寸: {w_orig}x{h_orig}")

    # 预处理
    input_size = (768, 1024)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_CUBIC)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 深度推理
    query_2d_uniform_coord = SAMPLING_METHODS["2d_uniform"](input_size).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_2d_uniform_depth, _ = model.inference(
            image=img_tensor,
            query_coord=query_2d_uniform_coord,
            gt_depth=None,
            gt_depth_mask=None,
            prompt_depth=None,
            prompt_mask=None,
        )

    depth_raw = pred_2d_uniform_depth.permute(0, 2, 1).view(1, 1, input_size[0], input_size[1])[0, 0]
    print(f"[InfiniDepth] 原始深度范围: [{depth_raw.min().item():.3f}, {depth_raw.max().item():.3f}] m")

    # 取倒数转 inverse depth
    inv_depth = 1.0 / depth_raw.clamp_min(1e-3)
    print(f"[InfiniDepth] Inverse 深度范围: [{inv_depth.min().item():.6f}, {inv_depth.max().item():.6f}]")

    # 归一化（和 DepthAnything 一样）
    flat = inv_depth.reshape(-1)
    sample_size = min(16384, flat.numel())
    idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=device))
    q_low, q_high = q_vals[0], q_vals[1]

    denom = (q_high - q_low).clamp_min(1e-6)
    depth_norm = ((inv_depth - q_low) / denom).clamp(0.0, 1.0)
    print(f"[InfiniDepth] 归一化后范围: [{depth_norm.min().item():.3f}, {depth_norm.max().item():.3f}]")

    # 保存可视化
    depth_np = depth_norm.cpu().numpy()
    depth_vis = (depth_np * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite("compare_infinidepth.png", depth_colormap)
    print(f"[InfiniDepth] 已保存: compare_infinidepth.png")

    del model
    torch.cuda.empty_cache()
    return depth_norm.cpu().numpy()


def test_depthanything(img_path, device):
    """测试 DepthAnythingV2 深度"""
    print(f"\n{'='*60}")
    print(f"[DepthAnythingV2] 测试: {Path(img_path).name}")
    print(f"{'='*60}")

    # 加载模型
    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitl.pth", map_location="cpu"))
    model = model.to(device).eval()

    # 加载图片
    img_bgr = cv2.imread(img_path)
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"[DepthAnything] 原始尺寸: {w_orig}x{h_orig}")

    # 预处理（和 mono2stereo_test.py 一致）
    input_size = 518
    long_edge = input_size
    scale = long_edge / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = long_edge
    else:
        depth_h = long_edge

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (depth_w, depth_h), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device).half()

    # 深度推理
    with torch.no_grad():
        depth = model(img_tensor)[0].float()

    print(f"[DepthAnything] 原始深度范围: [{depth.min().item():.3f}, {depth.max().item():.3f}]")

    # 归一化
    flat = depth.reshape(-1)
    sample_size = min(16384, flat.numel())
    idx = torch.randint(0, flat.numel(), (sample_size,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=device))
    q_low, q_high = q_vals[0], q_vals[1]

    denom = (q_high - q_low).clamp_min(1e-6)
    depth_norm = ((depth - q_low) / denom).clamp(0.0, 1.0)
    print(f"[DepthAnything] 归一化后范围: [{depth_norm.min().item():.3f}, {depth_norm.max().item():.3f}]")

    # 保存可视化
    depth_np = depth_norm.cpu().numpy()
    depth_vis = (depth_np * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite("compare_depthanything.png", depth_colormap)
    print(f"[DepthAnything] 已保存: compare_depthanything.png")

    del model
    torch.cuda.empty_cache()
    return depth_norm.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 找测试图片
    test_root = Path("/mnt/A/jiangxg/work/2Dto3D/submodules/InfiniDepth/example_huggingface")
    img_path = str(test_root / "eth3d_office_rgb.png")

    # 测试两个模型
    inf_depth = test_infinidepth(img_path, device)
    dav2_depth = test_depthanything(img_path, device)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"[总结] 两个模型的归一化深度对比")
    print(f"{'='*60}")
    print(f"InfiniDepth:    shape={inf_depth.shape}, mean={inf_depth.mean():.3f}, std={inf_depth.std():.3f}")
    print(f"DepthAnythingV2: shape={dav2_depth.shape}, mean={dav2_depth.mean():.3f}, std={dav2_depth.std():.3f}")
    print(f"\n两个模型现在都是 '值越大 = 越近' 的相对深度模式！")
    print(f"可以直接用于相同的视差计算逻辑。")


if __name__ == "__main__":
    main()
