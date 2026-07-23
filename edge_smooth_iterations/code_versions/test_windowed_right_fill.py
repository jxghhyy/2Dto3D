"""
测试新方案：只使用空洞右侧 N 像素范围内的背景像素
既不用左侧前景，也不用太远的背景
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

OUTPUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/debug_iterations")


def load_frame_at_70s():
    video_path = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()
    return frame_bgr


@torch.no_grad()
def compute_depth_and_hole(frame_bgr, device, max_disparity=24.0):
    h_orig, w_orig = frame_bgr.shape[:2]
    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    input_size = 518
    scale = input_size / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = input_size
    else:
        depth_h = input_size

    img_resized = F.interpolate(img, size=(depth_h, depth_w), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    model_input = (img_resized - mean_t) / std_t

    with torch.no_grad():
        depth_raw = model(model_input)[0].float()

    flat = depth_raw.reshape(-1)
    idx = torch.randint(0, flat.numel(), (16384,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=flat.device))
    low, high = q_vals[0], q_vals[1]
    depth_norm = ((depth_raw - low) / (high - low)).clamp(0.0, 1.0)

    dibr_h, dibr_w = h_orig, w_orig
    near_score = F.interpolate(
        depth_norm[None, None, :, :],
        size=(dibr_h, dibr_w),
        mode="bilinear", align_corners=False
    )[0, 0]
    disparity = near_score * max_disparity * dibr_w / w_orig

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    right_with_hole = right_warped.clone()
    right_with_hole[hole] = 0.0

    return left_rgb_tensor, right_with_hole, hole, near_score


def create_windowed_right_kernel(kernel_size, window_width, device):
    """
    创建窗口化的右侧卷积核
    window_width: 只使用中心右侧多少像素范围内的像素

    例如 kernel_size=21, window_width=10
    位置: -10 -9 ... -1 0 +1 +2 ... +10
    权重:   0  0      0 1  1      1   (只使用中心右侧0-10像素)
    """
    pad = kernel_size // 2
    distance_from_center = torch.arange(kernel_size, device=device) - pad

    # 只允许中心及其右侧 window_width 范围内的像素
    right_mask = (distance_from_center >= 0) & (distance_from_center <= window_width)
    weights = torch.zeros(kernel_size, device=device, dtype=torch.float32)
    weights[right_mask] = 1.0

    if weights.sum() > 1e-6:
        weights = weights / weights.sum() * right_mask.sum().float()

    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    return kernel_2d


def inpaint_windowed_right_with_debug(img, hole, kernel_size=21, window_width=10, max_iter=15):
    """
    窗口化右侧填补：每个空洞像素只参考其右侧 window_width 像素范围内的背景
    """
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_size // 2

    kernel = create_windowed_right_kernel(kernel_size, window_width, img.device)
    k3 = kernel.repeat(3, 1, 1, 1)

    steps = []

    for it in range(max_iter):
        bg_weight = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * bg_weight
        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, kernel, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
        filled_count = can_fill.sum().item()

        if filled_count == 0:
            break

        steps.append({
            "iter": it + 1,
            "image_before": result.clone(),
            "hole_before": hole_cur.clone(),
            "fill_mask": can_fill.clone(),
            "filled_count": filled_count,
            "method": f"windowed_w{window_width}"
        })

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        if not hole_cur.any():
            break

    steps.append({
        "iter": "final",
        "image_before": result.clone(),
        "hole_before": hole_cur.clone(),
        "fill_mask": torch.zeros_like(hole),
        "filled_count": 0,
        "method": f"windowed_w{window_width}_final"
    })

    return steps


def visualize_comparison(left_image, right_with_hole, hole, steps_v33, steps_v34, steps_win10, steps_win20):
    """生成最终对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h, w = hole.shape

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 第一行：原始图像
    axes[0, 0].imshow(left_image.cpu().numpy())
    axes[0, 0].set_title("左视图（原始）")

    axes[0, 1].imshow(right_with_hole.cpu().numpy())
    axes[0, 1].set_title(f"右视图（带空洞，{hole.sum().item():,} 像素）")

    hole_vis = np.zeros((h, w, 3), dtype=np.float32)
    hole_vis[hole.cpu().numpy(), 0] = 1.0
    axes[0, 2].imshow(hole_vis)
    axes[0, 2].set_title("空洞掩码（红色）")

    # 放大空洞区域的例子
    center_y, center_x = h // 2, w // 2 + 100
    crop_h, crop_w = 100, 150
    y1, y2 = max(0, center_y - crop_h//2), min(h, center_y + crop_h//2)
    x1, x2 = max(0, center_x - crop_w//2), min(w, center_x + crop_w//2)

    hole_crop = right_with_hole.cpu().numpy()[y1:y2, x1:x2]
    axes[0, 3].imshow(hole_crop)
    axes[0, 3].set_title("空洞区域放大")

    for ax in axes[0]:
        ax.axis('off')

    # 第二行：各种方法的结果
    methods = [
        ("v33 (偏置核)", steps_v33[-1]["image_before"]),
        ("v34 (单行扫描)", steps_v34[0]["image_before"]),
        ("v35 窗口 w=10", steps_win10[-1]["image_before"]),
        ("v35 窗口 w=20", steps_win20[-1]["image_before"]),
    ]

    for i, (name, img) in enumerate(methods):
        crop = img.cpu().numpy()[y1:y2, x1:x2]
        axes[1, i].imshow(crop)
        axes[1, i].set_title(name)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "00_method_comparison.png"), dpi=150)
    plt.close()

    # 迭代过程对比
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))

    all_steps = [
        ("v33", steps_v33),
        ("v34 scanline", steps_v34[:1] + [steps_v34[-1]]),
        ("v35 w=10", steps_win10),
        ("v35 w=20", steps_win20),
    ]

    for row_idx, (name, steps) in enumerate(all_steps):
        for col_idx, step in enumerate(steps[:6]):
            ax = axes[row_idx, col_idx]
            crop = step["image_before"].cpu().numpy()[y1:y2, x1:x2]

            # 标记本次填补的像素
            fill_crop = step["fill_mask"].cpu().numpy()[y1:y2, x1:x2]
            crop_marked = crop.copy()
            crop_marked[fill_crop, 0] = np.minimum(1.0, crop_marked[fill_crop, 0] + 0.5)

            ax.imshow(crop_marked)
            ax.set_title(f"{name} {step['iter']}\nfilled: {step['filled_count']:,}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "01_iteration_comparison.png"), dpi=150)
    plt.close()

    print(f"\n✅ 对比图已生成: {OUTPUT_DIR}")
    print(f"   - 00_method_comparison.png: 方法整体对比")
    print(f"   - 01_iteration_comparison.png: 迭代过程对比")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载帧...")
    frame_bgr = load_frame_at_70s()

    print("计算深度和空洞...")
    left_image, right_with_hole, hole, near_score = compute_depth_and_hole(frame_bgr, device, max_disparity=24)

    print(f"\nv33 方法 (11×11, bias=10)...")
    steps_v33 = inpaint_windowed_right_with_debug(  # 复用调试函数，用 window_width 很大来模拟 v33
        right_with_hole, hole, kernel_size=11, window_width=100, max_iter=10
    )
    for step in steps_v33[:-1]:
        print(f"  迭代 {step['iter']}: 填补 {step['filled_count']:,} 像素")

    print(f"\nv34 方法 (单行扫描)...")
    from debug_iteration_process import inpaint_v34_scanline_with_debug
    steps_v34 = inpaint_v34_scanline_with_debug(right_with_hole, hole, near_score, smooth_iterations=3)
    print(f"  1 次全部填补: {hole.sum().item():,} 像素")

    print(f"\nv35 窗口 w=10 (21×21 核，只使用右侧10像素)...")
    steps_win10 = inpaint_windowed_right_with_debug(
        right_with_hole, hole, kernel_size=21, window_width=10, max_iter=15
    )
    for step in steps_win10[:-1]:
        print(f"  迭代 {step['iter']}: 填补 {step['filled_count']:,} 像素")

    print(f"\nv35 窗口 w=20 (41×41 核，只使用右侧20像素)...")
    steps_win20 = inpaint_windowed_right_with_debug(
        right_with_hole, hole, kernel_size=41, window_width=20, max_iter=15
    )
    for step in steps_win20[:-1]:
        print(f"  迭代 {step['iter']}: 填补 {step['filled_count']:,} 像素")

    # 生成对比图
    visualize_comparison(left_image, right_with_hole, hole, steps_v33, steps_v34, steps_win10, steps_win20)


if __name__ == "__main__":
    main()
