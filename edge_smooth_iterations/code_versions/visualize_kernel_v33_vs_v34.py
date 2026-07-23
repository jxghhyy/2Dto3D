"""
可视化对比 v33 和 v34 的卷积核差异
验证 v34 是否真正排除了左侧像素的影响
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def create_v33_kernel(kernel_size, bias_strength):
    """v33 的偏置卷积核"""
    pad = kernel_size // 2
    weights = torch.exp(torch.linspace(0, np.log(bias_strength), kernel_size))
    weights = weights / weights.mean()
    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    return kernel_2d[0, 0].numpy()


def create_v34_kernel(kernel_size, right_bias_decay):
    """v34 的严格右侧卷积核"""
    pad = kernel_size // 2
    distance_from_center = torch.arange(kernel_size) - pad
    left_mask = distance_from_center < 0
    right_distances = distance_from_center.float()
    weights = torch.exp(-right_distances * right_bias_decay)
    weights[left_mask] = 0.0
    if weights.sum() > 1e-6:
        weights = weights / weights.sum() * (pad + 1)
    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    return kernel_2d[0, 0].numpy()


def visualize_kernels():
    kernel_size = 11
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # v33 不同偏置强度
    for i, bias in enumerate([1.0, 5.0, 10.0]):
        kernel = create_v33_kernel(kernel_size, bias)
        im = axes[0, i].imshow(kernel, cmap='viridis', vmin=0, vmax=kernel.max())
        axes[0, i].set_title(f'v33, bias={bias}x')
        axes[0, i].set_xlabel('水平位置（左→右）')
        axes[0, i].set_ylabel('垂直位置')
        plt.colorbar(im, ax=axes[0, i])

        # 打印水平中心线的权重
        center_row = kernel[kernel_size//2, :]
        print(f"v33 bias={bias}x 水平中心线权重:")
        print(f"  左侧 3 列: {center_row[:3]:.3f}, {center_row[3]:.3f}, {center_row[4]:.3f}")
        print(f"  中心列: {center_row[5]:.3f}")
        print(f"  右侧 3 列: {center_row[6]:.3f}, {center_row[7]:.3f}, {center_row[8]:.3f}")
        print()

    # v34 不同衰减因子
    for i, decay in enumerate([0.5, 1.0, 2.0]):
        kernel = create_v34_kernel(kernel_size, decay)
        im = axes[1, i].imshow(kernel, cmap='viridis', vmin=0, vmax=kernel.max())
        axes[1, i].set_title(f'v34, decay={decay}')
        axes[1, i].set_xlabel('水平位置（左→右）')
        axes[1, i].set_ylabel('垂直位置')
        plt.colorbar(im, ax=axes[1, i])

        # 打印水平中心线的权重
        center_row = kernel[kernel_size//2, :]
        print(f"v34 decay={decay} 水平中心线权重:")
        print(f"  左侧 3 列: {center_row[:3]:.3f}, {center_row[3]:.3f}, {center_row[4]:.3f}")
        print(f"  中心列: {center_row[5]:.3f}")
        print(f"  右侧 3 列: {center_row[6]:.3f}, {center_row[7]:.3f}, {center_row[8]:.3f}")
        print()

    plt.suptitle('v33 vs v34 卷积核对比（11×11）', fontsize=16)
    plt.tight_layout()
    plt.savefig('/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/kernel_comparison.png', dpi=150)
    print("✅ 对比图已保存: kernel_comparison.png")


def test_fill_direction():
    """测试在一个简单场景下的填充方向"""
    h, w = 10, 20

    # 创建一个测试图像：左侧红色前景，右侧蓝色背景
    img = torch.zeros(h, w, 3)
    img[:, :10, 0] = 1.0  # 左侧红色前景
    img[:, 10:, 2] = 1.0  # 右侧蓝色背景

    # 在中间创建一个空洞（模拟物体右移后的空洞）
    hole = torch.zeros(h, w, dtype=torch.bool)
    hole[:, 8:12] = True  # 空洞覆盖左（前景）右（背景）边缘

    print("测试场景:")
    print(f"  图像尺寸: {h}x{w}")
    print(f"  前景: 列 0-9 (红色)")
    print(f"  背景: 列 10-19 (蓝色)")
    print(f"  空洞: 列 8-11 (跨越前景和背景)")
    print()

    near = torch.ones(h, w) * 0.5  # 都算是背景

    # 模拟空洞区域
    img_with_hole = img.clone()
    img_with_hole[hole] = 0.0

    # 测试 v33 填充
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from iter_v33_right_to_left import fast_inpaint_v33_biased, create_biased_kernel
    from iter_v34_strict_right_only import fast_inpaint_v34_strict_right, create_strict_right_kernel

    device = torch.device('cpu')

    print("=" * 60)
    print("v33 测试 (bias_strength=10x):")
    print("-" * 60)

    # 显示 v33 卷积核
    v33_kernel = create_biased_kernel(7, 10.0, device, torch.float32)
    print("v33 7×7 核水平中心线权重:")
    print(v33_kernel[0, 0, 3, :].numpy())

    result_v33, _, _ = fast_inpaint_v33_biased(
        img_with_hole, hole, near,
        edge_kernel_size=7, inner_kernel_size=11,
        bias_strength=10.0, bg_threshold=0.5, max_iter=10
    )

    # 检查空洞区域的颜色
    hole_colors_v33 = result_v33[hole]
    avg_red = hole_colors_v33[:, 0].mean().item()
    avg_blue = hole_colors_v33[:, 2].mean().item()
    print(f"\n空洞区域平均颜色:")
    print(f"  红色分量 (来自前景): {avg_red:.3f}")
    print(f"  蓝色分量 (来自背景): {avg_blue:.3f}")
    print(f"  前景污染比例: {avg_red/(avg_red+avg_blue+1e-6)*100:.1f}%")

    print()
    print("=" * 60)
    print("v34 测试 (right_bias_decay=2.0):")
    print("-" * 60)

    # 显示 v34 卷积核
    v34_kernel = create_strict_right_kernel(7, 2.0, device, torch.float32)
    print("v34 7×7 核水平中心线权重:")
    print(v34_kernel[0, 0, 3, :].numpy())

    result_v34, _, _ = fast_inpaint_v34_strict_right(
        img_with_hole, hole, near,
        edge_kernel_size=7, inner_kernel_size=15,
        right_bias_decay=2.0, bg_threshold=0.5, max_iter=10
    )

    # 检查空洞区域的颜色
    hole_colors_v34 = result_v34[hole]
    avg_red = hole_colors_v34[:, 0].mean().item()
    avg_blue = hole_colors_v34[:, 2].mean().item()
    print(f"\n空洞区域平均颜色:")
    print(f"  红色分量 (来自前景): {avg_red:.3f}")
    print(f"  蓝色分量 (来自背景): {avg_blue:.3f}")
    print(f"  前景污染比例: {avg_red/(avg_red+avg_blue+1e-6)*100:.1f}%")

    # 可视化结果
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img.numpy())
    axes[0].set_title('原始图像')

    axes[1].imshow(img_with_hole.numpy())
    axes[1].set_title('带空洞的图像')

    axes[2].imshow(result_v33.numpy())
    axes[2].set_title('v33 填充结果')

    axes[3].imshow(result_v34.numpy())
    axes[3].set_title('v34 填充结果')

    for ax in axes:
        ax.axvline(x=9.5, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=7.5, color='red', linestyle=':', alpha=0.5)
        ax.axvline(x=11.5, color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/fill_direction_test.png', dpi=150)
    print("\n✅ 填充方向测试图已保存: fill_direction_test.png")


if __name__ == "__main__":
    print("🎯 v33 vs v34 卷积核对比测试\n")
    visualize_kernels()
    print("\n" + "=" * 60 + "\n")
    test_fill_direction()
    print("\n✅ 所有测试完成！")
