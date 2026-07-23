"""
迭代 v36: 方向感知的严格单侧卷积

核心改进：
1. ✅ 水平方向：严格右侧，左侧权重 = 0（彻底避免左侧前景渗入）
2. ✅ 垂直方向：动态检测上下方是否是前景，如果是则权重置 0
3. ✅ 核形状自适应：每个空洞像素的卷积核形状由周围前景分布决定
"""
import sys
import argparse
import subprocess
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


def create_strict_right_base_kernel(kernel_size, device):
    """
    基础严格右侧核（水平方向左侧为0，垂直方向初始均匀）
    """
    pad = kernel_size // 2
    x_indices = torch.arange(kernel_size, device=device) - pad

    # 水平方向：左侧 = 0，只保留中心及右侧
    horizontal_mask = x_indices >= 0
    horizontal_weights = torch.ones(kernel_size, device=device, dtype=torch.float32)
    horizontal_weights[~horizontal_mask] = 0.0

    # 2D 核：垂直方向初始为均匀
    kernel_1d = horizontal_weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)

    # 归一化：只计算非零权重的平均值
    num_nonzero = horizontal_weights.sum().item() * kernel_size
    if num_nonzero > 0:
        kernel_2d = kernel_2d / kernel_2d.sum() * num_nonzero

    return kernel_2d


@torch.no_grad()
def compute_foreground_probability(img, hole, near_score, bg_threshold=0.3):
    """
    估计每个非空洞像素是前景的概率
    简单启发式：近深度（near_score小）+ 在空洞左侧 → 概率高
    """
    h, w = hole.shape
    device = hole.device

    # 1. 深度线索：越近越可能是前景
    # near_score 越小 = 越近 = 越可能是前景
    depth_prob = 1.0 - near_score  # [0, 1]，近=1，远=0

    # 2. 位置线索：在空洞左侧的更可能是前景
    # 计算每行的左边缘位置
    hole_left_edge = torch.argmax(hole.float(), dim=1)  # [h]

    x_coords = torch.arange(w, device=device).view(1, w).expand(h, w)
    left_of_hole = x_coords < hole_left_edge.view(h, 1)

    # 综合概率
    fg_prob = depth_prob * left_of_hole.float()

    return fg_prob


@torch.no_grad()
def inpaint_v36_direction_aware(img, hole, near_score,
                                 kernel_size=15,
                                 max_iter=20,
                                 bg_threshold=0.3):
    """
    v36 方向感知填补

    对每个空洞像素：
    1. 水平方向：永远只用右侧像素
    2. 垂直方向：检查上下方是前景还是背景，如果是前景则排除
    """
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_size // 2
    device = hole.device
    h, w = hole.shape

    # 基础严格右侧核
    base_kernel = create_strict_right_base_kernel(kernel_size, device)

    # 先验前景概率（不变）
    fg_prob = compute_foreground_probability(img, hole, near_score, bg_threshold)

    # 用于缓存和调试
    steps = []

    for it in range(max_iter):
        # ========== 关键：为当前空洞区域动态调整卷积核 ==========
        # 我们用一个巧妙的方法：不用改变核本身，而是对被认为是前景的像素先置0

        # 创建前景掩码：非空洞 + 大概率是前景
        # 关键修正：near_score 越小越近（前景），越大越远（背景）！
        is_background = ~hole_cur & (near_score > bg_threshold)  # 远处 = 背景 ✓
        fg_mask = ~hole_cur & ~is_background  # 近处非空洞 = 前景 ✗ 需要屏蔽

        # 把前景像素在输入图像中置0，这样卷积时它们不会贡献颜色
        input_img = result.clone()
        input_img[fg_mask] = 0.0

        # 权重图：背景=1，前景=0，这样前景也不会贡献权重
        weight_map = is_background.float().unsqueeze(-1)

        # ========== 严格右侧卷积 ==========
        weighted_img = input_img * weight_map

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = weight_map.permute(2, 0, 1).unsqueeze(0)

        k3 = base_kernel.repeat(3, 1, 1, 1)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, base_kernel, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
        filled_count = can_fill.sum().item()

        if filled_count == 0:
            break

        # 记录步骤（用于调试）
        steps.append({
            "iter": it + 1,
            "image": result.clone(),
            "filled_count": filled_count,
            "remaining_hole": hole_cur.sum().item()
        })

        # 填补
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
    """调试模式：只处理第70秒帧，输出逐次处理效果"""
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v36] 使用设备: {device}")

    # 输出目录：vxx 风格
    OUTPUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v36")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[v36] 输出目录: {OUTPUT_DIR}")

    # ========== 加载模型 ==========
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # ========== 加载第70秒帧 ==========
    video_path = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    print(f"[v36] 视频帧率: {fps} fps")
    print(f"[v36] 处理第 70 秒 → 第 {target_frame} 帧")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("无法读取帧")

    h_orig, w_orig = frame_bgr.shape[:2]

    # ========== 深度推理 ==========
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

    print(f"[v36] 空洞像素数: {hole_with_band.sum().item()} ({hole_with_band.sum().item()/(w_orig*h_orig)*100:.2f}%)")

    # ========== v36 填补 ==========
    print(f"[v36] 开始方向感知填补 (核 15×15)...")
    t0 = time.time()
    result_v36, steps = inpaint_v36_direction_aware(
        right_with_band, hole_with_band, near_score,
        kernel_size=15, max_iter=20
    )
    t_v36 = time.time() - t0
    print(f"[v36] 填补完成，耗时: {t_v36*1000:.1f} ms")
    print(f"[v36] 实际迭代次数: {len(steps)-1} 次")

    # ========== 输出逐次处理的纯粹图片 ==========
    for i, step in enumerate(steps):
        img_np = (step["image"].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(OUTPUT_DIR / f"iter_{i:02d}_{step['iter']}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if i < len(steps) - 1:
            print(f"  迭代 {i+1}: 填补 {step['filled_count']:,} 像素，剩余 {step['remaining_hole']:,} 像素")

    # ========== 也输出 SBS 对比 ==========
    left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
    right_uint8 = (result_v36 * 255).byte().cpu().numpy()
    sbs = np.concatenate([left_uint8, right_uint8], axis=1)
    cv2.imwrite(str(OUTPUT_DIR / "final_sbs.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    print(f"\n[v36] ✅ 所有图片已输出到: {OUTPUT_DIR}")
    print(f"       文件: iter_00_1.png ~ iter_{len(steps)-1:02d}_final.png")


if __name__ == "__main__":
    main_debug_frame()
