#!/usr/bin/env python3
"""
简单的 InfiniDepth 单图深度推理测试
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submodules/InfiniDepth"))

import torch
import cv2
import numpy as np
from pathlib import Path

from InfiniDepth.utils.model_utils import build_model
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] 使用设备: {device}")

    # 加载模型
    model_path = "checkpoints/infinidepth.ckpt"
    print(f"[test] 加载 InfiniDepth 模型: {model_path}")

    model = build_model("InfiniDepth", model_path=model_path)
    model = model.to(device).eval()
    print("[test] 模型加载完成!")

    # 找一张测试图片
    test_root = Path("/mnt/A/jiangxg/work/2Dto3D/submodules/InfiniDepth/example_huggingface")
    test_images = list(test_root.rglob("*rgb*.png"))[:1]
    if not test_images:
        test_images = list(test_root.rglob("*.png"))[:1]

    if not test_images:
        print("[ERROR] 没有找到测试图片!")
        return

    img_path = test_images[0]
    print(f"[test] 测试图片: {img_path}")

    # 加载图片
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] 无法加载图片: {img_path}")
        return

    org_h, org_w = img_bgr.shape[:2]
    print(f"[test] 原始尺寸: {org_w}x{org_h}")

    # 预处理
    input_size = (512, 768)  # 小一点的尺寸加速测试
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_CUBIC)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 构建采样坐标
    print(f"[test] 构建采样坐标...")
    query_2d_uniform_coord = SAMPLING_METHODS["2d_uniform"](input_size).unsqueeze(0).to(device)

    # 深度推理
    print(f"[test] 开始深度推理...")
    with torch.no_grad():
        pred_2d_uniform_depth, _ = model.inference(
            image=img_tensor,
            query_coord=query_2d_uniform_coord,
            gt_depth=None,
            gt_depth_mask=None,
            prompt_depth=None,
            prompt_mask=None,
        )

    # 重塑深度图
    depth_raw = pred_2d_uniform_depth.permute(0, 2, 1).view(1, 1, input_size[0], input_size[1])[0, 0]
    print(f"[test] 深度范围: [{depth_raw.min().item():.3f}, {depth_raw.max().item():.3f}]")

    # 归一化并保存可视化
    depth_np = depth_raw.cpu().numpy()
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # 保存结果
    out_path = "test_infinidepth_depth.png"
    cv2.imwrite(out_path, depth_colormap)
    print(f"[test] 深度图已保存: {out_path}")

    print("[test] ✓ 测试成功!")


if __name__ == "__main__":
    main()
