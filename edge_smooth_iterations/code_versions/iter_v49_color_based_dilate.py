"""
迭代 v49: 基于颜色相似度的空洞膨胀算法

💡 核心思路：
  对于每个空洞的左侧边界像素，检测右侧最多 N 个像素
  如果颜色与左侧前景相似 → 也是前景残留，变成空洞
  如果不相似 → 遇到真实背景，停止膨胀

优势：
  1. 精确：逐像素验证，不是粗略估计
  2. GPU 并行：每行独立处理
  3. 提前终止：遇到不相似就停
  4. 解决边缘残留问题
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
BASE_OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v49_color_dilate")


def create_asymmetric_kernel(kernel_size, device, left_width=5):
    pad = kernel_size // 2
    x_indices = torch.arange(kernel_size, device=device) - pad
    horizontal_mask = x_indices >= -left_width
    horizontal_weights = torch.ones(kernel_size, device=device, dtype=torch.float32)
    horizontal_weights[~horizontal_mask] = 0.0
    kernel_1d = horizontal_weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    num_nonzero = horizontal_weights.sum().item() * kernel_size
    if num_nonzero > 0:
        kernel_2d = kernel_2d / kernel_2d.sum() * num_nonzero
    return kernel_2d


@torch.no_grad()
def dilate_by_color_similarity(right_warped, hole, max_dilate=8, color_threshold=0.05, device='cuda'):
    """
    基于颜色相似度的空洞膨胀算法

    原理：
    1. 找到每行空洞的左边界（hole[x-1]=False, hole[x]=True）
    2. 左边界左边的像素就是前景颜色 reference
    3. 向右检测最多 max_dilate 个像素
    4. 如果右边像素颜色 ≈ reference → 也是前景残留，变成空洞
    5. 遇到不相似就停止

    参数：
        max_dilate: 最多向右膨胀像素数（默认 8）
        color_threshold: RGB 颜色相似度阈值（0-1，默认 0.05 = 约13/255）
    """
    h, w = hole.shape
    hole_dilated = hole.clone()

    # ========== 第一步：找到所有空洞左边界 ==========
    # 左边界定义：当前是空洞，左边不是空洞
    hole_shift_left = F.pad(hole.float().unsqueeze(0).unsqueeze(0), (1, 0), value=0)[0, 0]
    left_boundary = hole & (~(hole_shift_left[:, :-1] > 0.5))

    # ========== 第二步：对每个左边界，获取参考颜色 ==========
    # 参考颜色 = 左边界左边 1 个像素
    ref_colors = torch.zeros((h, w, 3), device=device, dtype=torch.float32)
    for dy in range(h):
        boundary_x = torch.where(left_boundary[dy])[0]  # 该行所有左边界的 x 坐标
        if len(boundary_x) > 0:
            # 边界左边的像素颜色作为参考
            ref_x = torch.clamp(boundary_x - 1, 0, w - 1)
            ref_colors[dy, boundary_x] = right_warped[dy, ref_x]

    # ========== 第三步：逐次向右检测膨胀 ==========
    # 为每个边界位置记录：是否还在继续膨胀
    still_expanding = left_boundary.clone()
    current_ref_colors = ref_colors.clone()

    for shift in range(1, max_dilate + 1):
        # 还在膨胀的边界，检测 shift 位置的像素
        check_x = torch.where(still_expanding)[1] + shift
        check_y = torch.where(still_expanding)[0]

        # 超出右边界的不算
        valid = check_x < w
        if not valid.any():
            break

        cy = check_y[valid]
        cx = check_x[valid]

        # 该位置已经是空洞？跳过
        already_hole = hole_dilated[cy, cx]
        if already_hole.all():
            continue

        # 颜色相似度检测
        pixel_colors = right_warped[cy[~already_hole], cx[~already_hole]]
        ref_for_pixel = current_ref_colors[cy[~already_hole], cx[~already_hole] - shift]

        # 计算颜色差（L1 距离）
        color_diff = torch.abs(pixel_colors - ref_for_pixel).mean(dim=1)
        similar = color_diff < color_threshold

        # 相似 → 膨胀成空洞
        if similar.any():
            expand_y = cy[~already_hole][similar]
            expand_x = cx[~already_hole][similar]
            hole_dilated[expand_y, expand_x] = True

        # 不相似 → 停止这个边界的膨胀
        # 方法：更新 still_expanding
        still_expanding_np = still_expanding.cpu().numpy()
        sy = cy[~already_hole][~similar].cpu().numpy()
        sx = (cx[~already_hole][~similar] - shift).cpu().numpy()
        still_expanding_np[sy, sx] = False
        still_expanding = torch.from_numpy(still_expanding_np).to(device)

        if not still_expanding.any():
            break

    return hole_dilated


@torch.no_grad()
def dilate_by_color_similarity_fast(right_warped, hole, max_dilate=8, color_threshold=0.05, device='cuda'):
    """
    💡 最终简化版：基于颜色相似度的空洞膨胀算法

    原理（逐行处理）：
    1. 对每行，找到所有连续空洞区域 [start_x, end_x]
    2. 参考颜色 = 空洞左边的第一个非空洞像素（前景）
    3. 从 end_x + 1 开始，向右检测最多 max_dilate 个像素
    4. 如果像素颜色 ≈ 参考颜色 → 前景残留 → 变成空洞
    5. 遇到不相似的颜色 → 立即停止（提前终止）
    """
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()

    for y in range(h):
        row = hole_np[y]

        # 找到这行的所有连续空洞区域
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        # 分割成连续区域
        regions = []
        start_x = indices[0]
        prev_x = indices[0]
        for i in range(1, len(indices)):
            x = indices[i]
            if x > prev_x + 1:
                regions.append((start_x, prev_x))
                start_x = x
            prev_x = x
        regions.append((start_x, prev_x))

        # 对每个空洞区域处理
        for start_x, end_x in regions:
            # 参考颜色 = 空洞左边第一个非空洞像素
            if start_x > 0:
                ref_color = right_warped[y, start_x - 1].cpu().numpy()
            else:
                continue  # 空洞在最左边，没有参考颜色

            # 向右检测最多 max_dilate 个像素
            for shift in range(1, max_dilate + 1):
                check_x = end_x + shift
                if check_x >= w:
                    break

                # 已经是空洞了？跳过（理论上不应该）
                if hole_dilated[y, check_x]:
                    continue

                # 颜色相似度检测
                pixel_color = right_warped[y, check_x].cpu().numpy()
                color_diff = np.abs(pixel_color - ref_color).mean()

                if color_diff < color_threshold:
                    # 相似 → 前景残留，变成空洞
                    hole_dilated[y, check_x] = True
                else:
                    # 不相似 → 遇到真正背景，提前终止
                    break

    return torch.from_numpy(hole_dilated).to(device)


@torch.no_grad()
def inpaint_with_kernel(img, hole, kernel, k3, max_iter=10):
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2

    for it in range(max_iter):
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

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        if not hole_cur.any():
            break

    return result, hole_cur


def main_debug_frame():
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v49] 设备: {device}")
    print(f"[v49] 输出目录: {BASE_OUT_DIR}")

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # 加载第70秒帧
    video_path = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    h_orig, w_orig = frame_bgr.shape[:2]

    # 预处理
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

    near_score = F.interpolate(
        depth_norm[None, None, :, :],
        size=(h_orig, w_orig),
        mode="bilinear", align_corners=False
    )[0, 0]
    max_disparity = 24.0
    disparity = near_score * max_disparity

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    print(f"原始空洞: {hole.sum().item():,} 像素")

    # ========== v47: 旧的自适应膨胀 ==========
    print(f"\n【v47: 自适应膨胀 max=5】")
    t0 = time.time()
    hole_float = hole.float()
    cumsum_from_right = torch.cumsum(hole_float.flip(dims=[1]), dim=1).flip(dims=[1])
    hole_widths = cumsum_from_right * hole_float
    dilate_per_pixel = (hole_widths / 24.0 * 5).round().long()
    dilate_per_pixel = torch.clamp(dilate_per_pixel, 1, 5)
    hole_v47 = hole.clone()
    for shift in range(1, 5 + 1):
        shifted = torch.roll(hole, shifts=shift, dims=1)
        shifted[:, :shift] = False
        should_dilate = (dilate_per_pixel >= shift) & shifted
        hole_v47 = hole_v47 | should_dilate
    torch.cuda.synchronize()
    t_v47 = (time.time() - t0) * 1000
    print(f"  耗时: {t_v47:.1f} ms")
    print(f"  膨胀后: {hole_v47.sum().item():,} 像素")

    # ========== v49: 基于颜色的膨胀 ==========
    print(f"\n【v49: 颜色相似度膨胀】")
    for thresh in [0.02, 0.05, 0.08, 0.1]:
        t0 = time.time()
        hole_v49 = dilate_by_color_similarity_fast(
            right_warped, hole, max_dilate=8, color_threshold=thresh, device=device
        )
        torch.cuda.synchronize()
        t_v49 = (time.time() - t0) * 1000
        new_pixels = hole_v49.sum().item() - hole.sum().item()
        print(f"  阈值={thresh:.2f}: {t_v49:.1f} ms, "
              f"膨胀后: {hole_v49.sum().item():,} (+{new_pixels})")

    # 用阈值 0.05 生成可视化
    hole_v49 = dilate_by_color_similarity_fast(
        right_warped, hole, max_dilate=8, color_threshold=0.05, device=device
    )

    # ========== 可视化对比 ==========
    print(f"\n【生成可视化对比...】")

    # 1. 原始空洞
    vis_before = (right_warped * 255).byte().cpu().numpy().copy()
    vis_before[hole.cpu().numpy()] = 0
    cv2.imwrite(str(BASE_OUT_DIR / "01_hole_original.png"),
               cv2.cvtColor(vis_before, cv2.COLOR_RGB2BGR))

    # 2. v47 膨胀结果
    vis_v47 = (right_warped * 255).byte().cpu().numpy().copy()
    vis_v47[hole_v47.cpu().numpy()] = 0
    cv2.imwrite(str(BASE_OUT_DIR / "02_hole_v47_adaptive5.png"),
               cv2.cvtColor(vis_v47, cv2.COLOR_RGB2BGR))

    # 3. v49 膨胀结果
    vis_v49 = (right_warped * 255).byte().cpu().numpy().copy()
    vis_v49[hole_v49.cpu().numpy()] = 0
    cv2.imwrite(str(BASE_OUT_DIR / "03_hole_v49_color_th05.png"),
               cv2.cvtColor(vis_v49, cv2.COLOR_RGB2BGR))

    # 4. 差异高亮：v49 - v47（v49 多膨胀的区域）
    diff_v49_more = hole_v49.cpu().numpy() & ~hole_v47.cpu().numpy()
    diff_v47_more = hole_v47.cpu().numpy() & ~hole_v49.cpu().numpy()

    vis_diff = (right_warped * 255).byte().cpu().numpy().copy()
    vis_diff[diff_v49_more] = [0, 255, 0]  # 绿色 = v49 多膨胀的（v47 漏了的）
    vis_diff[diff_v47_more] = [255, 0, 0]  # 红色 = v47 多膨胀的（v49 觉得不需要）
    cv2.imwrite(str(BASE_OUT_DIR / "04_diff_v49_vs_v47.png"),
               cv2.cvtColor(vis_diff, cv2.COLOR_RGB2BGR))

    print(f"  绿色 = v49 新增膨胀（v47 漏了的前景残留）")
    print(f"  红色 = v47 误膨胀（v49 认为是背景，不需要）")
    print(f"  v49 比 v47 多膨胀: {diff_v49_more.sum():,} 像素")
    print(f"  v47 比 v49 多膨胀: {diff_v47_more.sum():,} 像素")

    # 5. 裁剪放大人物边缘
    y1, y2 = 200, 700
    x1, x2 = 700, 1000

    for name, img_data in [
        ("05_crop_before.png", vis_before),
        ("06_crop_v47.png", vis_v47),
        ("07_crop_v49.png", vis_v49),
        ("08_crop_diff.png", vis_diff),
    ]:
        crop = img_data[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        crop_x2 = cv2.resize(crop_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(BASE_OUT_DIR / name), crop_x2)

    # ========== 测试填充效果 ==========
    print(f"\n【测试不对称核填充 (left_width=5, max_iter=8)...】")
    kernel = create_asymmetric_kernel(15, device, left_width=5)
    k3 = kernel.repeat(3, 1, 1, 1)

    right_with_hole = right_warped.clone()
    right_with_hole[hole_v49] = 0.0

    t0 = time.time()
    result, hole_remaining = inpaint_with_kernel(
        right_with_hole, hole_v49, kernel, k3, max_iter=8
    )
    torch.cuda.synchronize()
    t_inpaint = (time.time() - t0) * 1000

    print(f"  填充耗时: {t_inpaint:.1f} ms")
    print(f"  剩余空洞: {hole_remaining.item():,} 像素")

    # 保存填充结果
    result_np = (result * 255).byte().cpu().numpy()
    cv2.imwrite(str(BASE_OUT_DIR / "09_result_v49_inpaint.png"),
               cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

    # SBS 对比
    left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
    sbs = np.concatenate([left_uint8, result_np], axis=1)
    cv2.imwrite(str(BASE_OUT_DIR / "10_final_sbs.png"),
               cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    print(f"\n✅ v49 单帧测试完成！")
    print(f"  输出目录: {BASE_OUT_DIR}")


if __name__ == "__main__":
    main_debug_frame()
