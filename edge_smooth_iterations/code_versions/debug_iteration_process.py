"""
调试每一次空洞填补迭代的过程
显示：每次迭代填补了哪些像素，以及颜色变化
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_frame_at_70s():
    """加载第 70 秒的帧（25fps → 第 1750 帧）"""
    video_path = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_frame = int(70 * fps)
    print(f"视频帧率: {fps} fps")
    print(f"第 70 秒 → 第 {target_frame} 帧 (共 {total_frames} 帧)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("无法读取帧")

    return frame_bgr


@torch.no_grad()
def compute_depth_and_hole(frame_bgr, device, max_disparity=24.0):
    """计算深度、DIBR 扭曲，得到空洞"""
    h_orig, w_orig = frame_bgr.shape[:2]

    # 深度推理
    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

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

    print(f"空洞像素数: {hole.sum().item()} ({hole.sum().item()/(w_orig*h_orig)*100:.2f}%)")

    return left_rgb_tensor, right_with_hole, hole, near_score


def create_v33_kernel(kernel_size, bias_strength, device):
    """v33 偏置卷积核"""
    pad = kernel_size // 2
    weights = torch.exp(torch.linspace(0, np.log(bias_strength), kernel_size, device=device))
    weights = weights / weights.mean()
    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    return kernel_2d


def create_strict_right_kernel(kernel_size, device):
    """严格右侧卷积核（左半侧为0）"""
    pad = kernel_size // 2
    distance_from_center = torch.arange(kernel_size, device=device) - pad
    left_mask = distance_from_center < 0
    weights = torch.ones(kernel_size, device=device, dtype=torch.float32)
    weights[left_mask] = 0.0
    if weights.sum() > 1e-6:
        weights = weights / weights.sum() * (pad + 1)
    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    return kernel_2d


def inpaint_v33_with_debug(img, hole, near, kernel_size=11, bias_strength=10.0, max_iter=10):
    """v33 填补，返回每一步的中间结果"""
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_size // 2

    kernel = create_v33_kernel(kernel_size, bias_strength, img.device)
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

        # 记录这一步填补前的状态（用于对比）
        hole_mask = hole_cur.float()
        steps.append({
            "iter": it + 1,
            "image_before": result.clone(),
            "hole_before": hole_cur.clone(),
            "fill_mask": can_fill.clone(),
            "filled_count": filled_count,
            "method": "v33"
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
        "method": "v33"
    })

    return steps


def inpaint_v34_scanline_with_debug(img, hole, near, smooth_kernel_size=7, smooth_iterations=3):
    """v34 单行扫描 + 平滑，返回每一步的中间结果"""
    h, w = hole.shape
    steps = []

    # 步骤1: 单行扫描初始化
    img_flipped = img.flip(dims=[1])
    hole_flipped = hole.flip(dims=[1])
    x_indices = torch.arange(w, device=hole.device).view(1, w).expand(h, w)
    last_valid_pos = torch.where(~hole_flipped, x_indices, torch.tensor(-1, device=hole.device))
    last_valid_pos, _ = last_valid_pos.cummax(dim=1)
    last_valid_pos[last_valid_pos == -1] = 0
    y_indices = torch.arange(h, device=hole.device).view(h, 1).expand(h, w)
    result_flipped = img_flipped[y_indices, last_valid_pos, :]
    result = result_flipped.flip(dims=[1])

    steps.append({
        "iter": "scanline",
        "image_before": result.clone(),
        "hole_before": hole.clone(),
        "fill_mask": hole.clone(),
        "filled_count": hole.sum().item(),
        "method": "v34_scanline"
    })

    # 步骤2: 严格右侧卷积平滑
    pad = smooth_kernel_size // 2
    kernel = create_strict_right_kernel(smooth_kernel_size, hole.device)
    k3 = kernel.repeat(3, 1, 1, 1)
    hole_cur = hole.clone()

    for it in range(smooth_iterations):
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
            "iter": f"smooth_{it+1}",
            "image_before": result.clone(),
            "hole_before": hole_cur.clone(),
            "fill_mask": can_fill.clone(),
            "filled_count": filled_count,
            "method": "v34_smooth"
        })

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

    steps.append({
        "iter": "final",
        "image_before": result.clone(),
        "hole_before": hole_cur.clone(),
        "fill_mask": torch.zeros_like(hole),
        "filled_count": 0,
        "method": "v34_final"
    })

    return steps


def visualize_steps(steps, left_image, right_with_hole, hole, prefix):
    """可视化每一步的结果"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h, w = hole.shape

    # 1. 保存原始空洞和左图
    left_np = (left_image.cpu().numpy() * 255).astype(np.uint8)
    right_with_hole_np = (right_with_hole.cpu().numpy() * 255).astype(np.uint8)
    hole_np = hole.cpu().numpy().astype(np.uint8) * 255

    cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_00_left.png"), cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_01_right_with_hole.png"), cv2.cvtColor(right_with_hole_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_02_hole_mask.png"), hole_np)

    # 2. 保存每一步迭代
    for i, step in enumerate(steps):
        img_np = (step["image_before"].cpu().numpy() * 255).astype(np.uint8)

        # 在图像上用红色高亮标记本次填补的像素
        fill_mask_np = step["fill_mask"].cpu().numpy()
        img_marked = img_np.copy()
        img_marked[fill_mask_np, 0] = 255  # R
        img_marked[fill_mask_np, 1] = 0    # G
        img_marked[fill_mask_np, 2] = 0    # B

        # 空洞半透明蓝色覆盖
        hole_np_step = step["hole_before"].cpu().numpy()
        img_marked[hole_np_step, 2] = np.minimum(255, img_marked[hole_np_step, 2] + 128)

        cv2.imwrite(str(OUTPUT_DIR / f"{prefix}_{i+3:02d}_iter_{step['iter']}.png"), cv2.cvtColor(img_marked, cv2.COLOR_RGB2BGR))

        print(f"  {prefix} iter {step['iter']}: 填补了 {step['filled_count']} 像素")

    # 3. 拼接大图
    n_cols = 4
    n_rows = (len(steps) + n_cols - 1) // n_cols
    fig_h, fig_w = h // 4, w // 4  # 缩小显示

    plt.figure(figsize=(16, 4 * n_rows))

    for i, step in enumerate(steps):
        plt.subplot(n_rows, n_cols, i + 1)
        img_np = (step["image_before"].cpu().numpy() * 255).astype(np.uint8)
        img_marked = cv2.resize(img_np, (fig_w, fig_h))

        # 也标记填补区域
        fill_mask_small = cv2.resize(step["fill_mask"].cpu().numpy().astype(np.uint8), (fig_w, fig_h)) > 0
        img_marked[fill_mask_small, 0] = 255

        plt.imshow(img_marked)
        plt.title(f"{step['method']} - {step['iter']}\nfilled: {step['filled_count']:,}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"{prefix}_all_steps.png"), dpi=150)
    plt.close()

    print(f"\n  {prefix} 结果保存在: {OUTPUT_DIR}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print()

    # 1. 加载帧
    print("1. 加载第 70 秒的帧")
    frame_bgr = load_frame_at_70s()
    print()

    # 2. 计算深度和空洞
    print("2. 计算深度和空洞")
    left_image, right_with_hole, hole, near_score = compute_depth_and_hole(frame_bgr, device, max_disparity=24)
    print()

    # 3. v33 方法
    print("3. v33 方法（偏置卷积核）")
    steps_v33 = inpaint_v33_with_debug(right_with_hole, hole, near_score,
                                        kernel_size=11, bias_strength=10.0, max_iter=10)
    visualize_steps(steps_v33, left_image, right_with_hole, hole, "v33")
    print()

    # 4. v34 方法
    print("4. v34 方法（单行扫描 + 严格右侧平滑）")
    steps_v34 = inpaint_v34_scanline_with_debug(right_with_hole, hole, near_score,
                                                 smooth_kernel_size=7, smooth_iterations=5)
    visualize_steps(steps_v34, left_image, right_with_hole, hole, "v34")

    print()
    print("=" * 70)
    print(f"✅ 完成！所有调试图像保存在: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
