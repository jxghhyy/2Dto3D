"""
迭代 v36b: 只严格限制水平左侧，垂直方向正常

简化思路：
1. ✅ 水平方向：左侧权重严格 = 0（彻底防左侧前景渗入）
2. ✅ 垂直方向：不做特殊屏蔽（垂直方向的上下通常是背景或同种颜色）

羽毛球场景验证：
- 物体轮廓主要是垂直的 → 左侧是主要污染源
- 上下方通常也是草地（背景），颜色一致不会造成明显污染
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


def create_strict_right_kernel(kernel_size, device):
    """
    严格右侧卷积核
    水平方向：左侧所有像素权重 = 0，中心及右侧权重 = 1
    垂直方向：全部权重 = 1

    例如 kernel_size=15 (pad=7):
    水平位置: -7 -6 -5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5 +6 +7
    权重:      0  0  0  0  0  0  0  1  1  1  1  1  1  1  1

    这样从数学上保证：左侧像素永远不会参与计算！
    """
    pad = kernel_size // 2
    x_indices = torch.arange(kernel_size, device=device) - pad

    # 水平掩码：只允许中心及右侧
    horizontal_mask = x_indices >= 0
    horizontal_weights = torch.ones(kernel_size, device=device, dtype=torch.float32)
    horizontal_weights[~horizontal_mask] = 0.0

    # 2D 核
    kernel_1d = horizontal_weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)  # 垂直方向全部为1

    # 归一化：有效像素数量 = (pad + 1) * kernel_size
    num_nonzero = horizontal_weights.sum().item() * kernel_size
    if num_nonzero > 0:
        kernel_2d = kernel_2d / kernel_2d.sum() * num_nonzero

    return kernel_2d


@torch.no_grad()
def inpaint_v36b_strict_right(img, hole, kernel_size=15, max_iter=20):
    """
    v36b: 只严格限制水平左侧，垂直方向自由填充
    """
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_size // 2
    device = hole.device

    # 严格右侧卷积核（只创建一次）
    kernel = create_strict_right_kernel(kernel_size, device)
    k3 = kernel.repeat(3, 1, 1, 1)

    steps = []

    for it in range(max_iter):
        # 只使用已填充区域（非空洞）作为有效输入
        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

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
            "image": result.clone(),
            "filled_count": filled_count,
            "remaining_hole": hole_cur.sum().item()
        })

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        if not hole_cur.any():
            break

    steps.append({
        "iter": "final",
        "image": result.clone(),
        "filled_count": 0,
        "remaining_hole": hole_cur.sum().item()
    })

    return result, steps


@torch.no_grad()
def project_disocclusion_bands_gpu(disparity, min_drop=3.0, min_band_width=8):
    h, w = disparity.shape
    device = disparity.device

    disp_flipped = torch.flip(disparity, dims=[1])
    max_from_right = torch.cummax(disp_flipped, dim=1)[0]
    max_from_right = torch.flip(max_from_right, dims=[1])

    max_right_shifted = torch.roll(max_from_right, shifts=1, dims=1)
    max_right_shifted[:, 0] = 0.0

    drop_mask = disparity > (max_right_shifted + min_drop)
    band_length = disparity.clamp(min=0).long()

    diff = torch.zeros((h, w + 1), dtype=torch.int32, device=device)

    rows, cols = torch.where(drop_mask)
    if len(rows) > 0:
        starts = (cols - band_length[rows, cols] + 1).clamp(min=0)
        ends = cols.clamp(max=w - 1)
        valid_mask = (ends - starts) >= min_band_width
        if valid_mask.any():
            diff[rows[valid_mask], starts[valid_mask]] += 1
            diff[rows[valid_mask], ends[valid_mask] + 1] -= 1

    bands = torch.cumsum(diff[:, :w], dim=1) > 0
    return bands


def main_debug_frame():
    """调试模式：只处理第70秒帧"""
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v36b] 使用设备: {device}")

    OUTPUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v36b")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[v36b] 输出目录: {OUTPUT_DIR}")

    # 加载模型
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # 加载第70秒帧
    video_path = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    print(f"[v36b] 处理第 70 秒 → 第 {target_frame} 帧")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

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
    max_disparity = 24.0 * dibr_w / w_orig
    disparity = near_score * max_disparity

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    # 反遮挡带
    disocclusion_band = project_disocclusion_bands_gpu(disparity_sharp, min_drop=3.5, min_band_width=8)
    hole_with_band = hole | disocclusion_band
    right_with_band = right_warped.clone()
    right_with_band[hole_with_band] = 0.0

    print(f"[v36b] 空洞像素数: {hole_with_band.sum().item()} ({hole_with_band.sum().item()/(w_orig*h_orig)*100:.2f}%)")

    # 测试不同核尺寸
    for kernel_size in [11, 15, 21]:
        subdir = OUTPUT_DIR / f"kernel_{kernel_size}"
        subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n[v36b] 核 {kernel_size}×{kernel_size} 测试...")
        t0 = time.time()
        result, steps = inpaint_v36b_strict_right(
            right_with_band, hole_with_band, kernel_size=kernel_size, max_iter=20
        )
        t_v36 = time.time() - t0
        print(f"       耗时: {t_v36*1000:.1f} ms, {len(steps)-1} 次迭代")
        print(f"       剩余空洞: {steps[-1]['remaining_hole']:,} 像素")

        # 输出逐次图片
        for i, step in enumerate(steps):
            img_np = (step["image"].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(subdir / f"iter_{i:02d}_{step['iter']}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # 输出 SBS
        left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
        right_uint8 = (result * 255).byte().cpu().numpy()
        sbs = np.concatenate([left_uint8, right_uint8], axis=1)
        cv2.imwrite(str(subdir / "final_sbs.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    print(f"\n[v36b] ✅ 所有测试完成！输出到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main_debug_frame()
