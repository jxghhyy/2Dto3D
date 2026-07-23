"""
v56: 自适应卷积核大小 + 动态方向选择

特性：
1. 每个连通空洞区域独立分析其平均横向宽度
2. 从右向左填充：核大小自适应（宽度大 → 初始核大）
   - 初始核 = min(11, max(3, 平均宽度 // 2))
   - 每轮迭代后核尺寸减小（-2），直到 3
3. 左侧填充严格控制次数（最多3次）← 防止前景扩增
4. 检测右侧无像素的区域 → 自动切换从左向右，不设迭代上限
5. 只测试第20秒和第70秒帧
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
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v56_adaptive")

# 参数
MAX_DILATE = 8
DILATE_THRESHOLD = 0.15
MAX_LEFT_ITER = 3  # 从左向右最多3次（防止前景扩增）


def analyze_hole_regions(hole_np):
    """分析所有连通空洞区域，返回每个区域的信息"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hole_np, connectivity=8)

    regions = []
    for idx in range(1, num_labels):  # 跳过背景
        left = stats[idx, cv2.CC_STAT_LEFT]
        top = stats[idx, cv2.CC_STAT_TOP]
        width = stats[idx, cv2.CC_STAT_WIDTH]
        height = stats[idx, cv2.CC_STAT_HEIGHT]

        # 逐行计算实际平均宽度
        row_widths = []
        right_has_pixel = []

        for y in range(top, top + height):
            row = labels[y, left:left + width]
            hole_mask = (row == idx)

            if np.any(hole_mask):
                hole_cols = np.where(hole_mask)[0] + left
                min_x = hole_cols[0]
                max_x = hole_cols[-1]
                actual_width = max_x - min_x + 1
                row_widths.append(actual_width)

                # 检查右侧是否有像素
                if max_x + 1 < hole_np.shape[1]:
                    right_has_pixel.append(not hole_np[y, max_x + 1])
                else:
                    right_has_pixel.append(False)

        avg_width = np.mean(row_widths) if row_widths else 0
        right_valid_ratio = np.mean(right_has_pixel) if right_has_pixel else 0

        regions.append({
            'label': idx,
            'left': left,
            'top': top,
            'width': width,
            'height': height,
            'avg_row_width': avg_width,
            'right_valid_ratio': right_valid_ratio,
            'can_fill_right_to_left': right_valid_ratio > 0.5  # 超过一半行右侧有像素
        })

    return regions, labels


def create_directional_kernel(kernel_size, direction, device):
    """
    创建方向卷积核
    direction: 'right_to_left' = 只看右侧（向左填充）
               'left_to_right' = 只看左侧（向右填充）
    """
    half = kernel_size // 2
    kernel = torch.zeros((kernel_size, kernel_size), device=device)

    if direction == 'right_to_left':
        # 从右向左填充：看中心右侧的像素
        kernel[:, half:] = 1.0  # 包括中心和右侧
    elif direction == 'left_to_right':
        # 从左向右填充：看中心左侧的像素
        kernel[:, :half + 1] = 1.0  # 包括中心和左侧

    # 归一化
    kernel = kernel / kernel.sum()
    kernel_2d = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel_2d


@torch.no_grad()
def fill_hole_adaptive(img, hole, device):
    """
    自适应填充：
    - 对每个连通区域，根据平均宽度选择初始卷积核大小
    - 从右向左填充，核逐渐减小
    - 右侧无像素的区域，从左向右填充（次数不限）
    """
    result = img.clone()
    hole_cur = hole.clone()
    h, w = hole.shape

    hole_np = hole.cpu().numpy().astype(np.uint8) * 255

    # 分析所有空洞区域
    regions, labels_np = analyze_hole_regions(hole_np)
    labels = torch.from_numpy(labels_np).to(device)

    print(f"\n  空洞区域分析: 共 {len(regions)} 个区域")
    for i, r in enumerate(regions[:10]):
        print(f"    区域{r['label']}: 平均宽度={r['avg_row_width']:.1f}像素, "
              f"右侧有效={r['right_valid_ratio']*100:.0f}%, "
              f"方向={'右→左' if r['can_fill_right_to_left'] else '左→右'}")
    if len(regions) > 10:
        print(f"    ... 还有 {len(regions) - 10} 个小区域")

    # ========== 阶段1: 从右向左填充所有"右侧有像素"的区域 ==========
    print(f"\n  阶段1: 从右向左填充（核自适应）")

    # 为每个区域计算合适的初始核大小
    for r in regions:
        if not r['can_fill_right_to_left']:
            continue

        # 根据平均宽度设置初始核大小
        avg_w = r['avg_row_width']
        initial_kernel = min(11, max(3, int(avg_w // 2)))
        # 确保是奇数
        if initial_kernel % 2 == 0:
            initial_kernel += 1

        r['initial_kernel'] = initial_kernel
        r['remaining_iter'] = int(avg_w / (initial_kernel // 2)) + 1

        print(f"    区域{r['label']}: 初始核={initial_kernel}, "
              f"预计迭代={r['remaining_iter']}次")

    # 按区域分别填充（因为每个区域核大小不同）
    for r in regions:
        if not r['can_fill_right_to_left']:
            continue

        region_mask = (labels == r['label']) & hole_cur
        if not region_mask.any():
            continue

        kernel_size = r['initial_kernel']
        max_iter = r['remaining_iter']

        for it in range(max_iter):
            if kernel_size < 3:
                kernel_size = 3

            kernel = create_directional_kernel(kernel_size, 'right_to_left', device)
            k3 = kernel.repeat(3, 1, 1, 1)
            pad = kernel_size // 2

            valid_mask = (~hole_cur).unsqueeze(-1).float()
            weighted_img = result * valid_mask

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
            weight_sum = F.conv2d(weight_nchw, kernel, padding=pad)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = region_mask & hole_cur & (weight_sum[0, 0] > 0.5)
            filled = can_fill.sum().item()

            if filled == 0:
                break

            result[can_fill] = avg_hwc[can_fill]
            hole_cur[can_fill] = False
            region_mask[can_fill] = False

            # 核逐渐减小
            kernel_size -= 2

        print(f"    区域{r['label']}: 实际迭代{it+1}次, 剩余空洞={region_mask.sum().item()}")

    # ========== 阶段2: 从左向右填充"右侧无像素"的区域 ==========
    print(f"\n  阶段2: 从左向右填充（右侧无像素的区域，不设上限）")

    for r in regions:
        if r['can_fill_right_to_left']:
            continue

        region_mask = (labels == r['label']) & hole_cur
        if not region_mask.any():
            continue

        print(f"    区域{r['label']}: 从左向右填充（核=3，无上限）")

        it = 0
        while True:
            kernel = create_directional_kernel(3, 'left_to_right', device)
            k3 = kernel.repeat(3, 1, 1, 1)

            valid_mask = (~hole_cur).unsqueeze(-1).float()
            weighted_img = result * valid_mask

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3, padding=1, groups=3)
            weight_sum = F.conv2d(weight_nchw, kernel, padding=1)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = region_mask & hole_cur & (weight_sum[0, 0] > 0.5)
            filled = can_fill.sum().item()

            if filled == 0:
                break

            result[can_fill] = avg_hwc[can_fill]
            hole_cur[can_fill] = False
            region_mask[can_fill] = False
            it += 1

        print(f"    区域{r['label']}: 实际迭代{it}次, 剩余空洞={region_mask.sum().item()}")

    # ========== 阶段3: 全局收尾填充（防止区域间缝隙）==========
    print(f"\n  阶段3: 全局收尾填充")
    kernel = create_directional_kernel(3, 'right_to_left', device)
    k3 = kernel.repeat(3, 1, 1, 1)

    for it in range(3):
        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=1, groups=3)
        weight_sum = F.conv2d(weight_nchw, kernel, padding=1)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
        filled = can_fill.sum().item()

        if filled == 0:
            break

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

    print(f"    收尾填充: 迭代{it+1}次, 最终剩余={hole_cur.sum().item()}")

    return result, hole_cur


@torch.no_grad()
def dilate_right_gpu_simple(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()
    right_warped_np = right_warped.cpu().numpy()
    th = color_threshold * 255

    for y in range(h):
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        if len(indices) == 1:
            regions = [(indices[0], indices[0])]
        else:
            diff = np.diff(indices)
            splits = np.where(diff > 1)[0] + 1
            if len(splits) == 0:
                regions = [(indices[0], indices[-1])]
            else:
                regions = []
                prev = 0
                for s in splits:
                    regions.append((indices[prev], indices[s-1]))
                    prev = s
                regions.append((indices[prev], indices[-1]))

        for start_x, end_x in regions:
            if start_x <= 0:
                continue
            ref_color = right_warped_np[y, start_x - 1]

            for shift in range(1, max_dilate + 1):
                check_x = end_x + shift
                if check_x >= w:
                    break
                if hole_dilated[y, check_x]:
                    continue

                pixel_color = right_warped_np[y, check_x]
                color_diff = np.abs(pixel_color.astype(float) - ref_color.astype(float)).mean()

                if color_diff < th:
                    hole_dilated[y, check_x] = True
                else:
                    break

    return torch.from_numpy(hole_dilated).to(device)


def process_frame(frame_idx, frame_bgr, model, device, frame_name):
    """处理单帧"""
    h, w = frame_bgr.shape[:2]

    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

    input_size = 518
    scale = input_size / max(h, w)
    depth_h = max(14, int(round(h * scale / 14)) * 14)
    depth_w = max(14, int(round(w * scale / 14)) * 14)
    if w >= h:
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
        size=(h, w),
        mode="bilinear", align_corners=False
    )[0, 0]
    disparity = near_score * 24.0

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    print(f"\n原始空洞像素: {hole.sum().item()}")

    # 向右膨胀（检测白色抗锯齿像素）
    hole_dilated = dilate_right_gpu_simple(
        right_warped, hole, MAX_DILATE, DILATE_THRESHOLD, device
    )
    print(f"膨胀后空洞像素: {hole_dilated.sum().item()}")

    right_with_hole = right_warped.clone()
    right_with_hole[hole_dilated] = 0.0

    # v56 自适应填充
    result, hole_final = fill_hole_adaptive(right_with_hole, hole_dilated, device)

    result_np = (result.cpu().numpy() * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    # 剩余空洞可视化
    remaining_viz = result_bgr.copy()
    remaining_mask = hole_final.cpu().numpy()
    remaining_viz[remaining_mask] = [0, 0, 255]  # 红色标记剩余空洞

    # 拼接对比图
    original_right = (right_warped.cpu().numpy() * 255).astype(np.uint8)
    original_right_bgr = cv2.cvtColor(original_right, cv2.COLOR_RGB2BGR)

    combined = np.hstack([frame_bgr, original_right_bgr, result_bgr, remaining_viz])

    cv2.imwrite(str(OUT_DIR / f"{frame_name}_result.png"), result_bgr)
    cv2.imwrite(str(OUT_DIR / f"{frame_name}_combined.png"), combined)

    print(f"\n结果已保存: {frame_name}_result.png / {frame_name}_combined.png")
    return result_bgr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v56 自适应卷积核] 设备: {device}")
    print(f"[v56 自适应卷积核] 输出目录: {OUT_DIR}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {fps:.1f} fps, 共 {total_frames} 帧")

    # 处理第20秒
    print("\n" + "="*60)
    print("处理第20秒")
    print("="*60)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(20 * fps))
    ok, frame_bgr = cap.read()
    if ok:
        process_frame(int(20 * fps), frame_bgr, model, device, "frame_20s")

    # 处理第70秒
    print("\n" + "="*60)
    print("处理第70秒")
    print("="*60)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(70 * fps))
    ok, frame_bgr = cap.read()
    if ok:
        process_frame(int(70 * fps), frame_bgr, model, device, "frame_70s")

    cap.release()
    print("\n[v56] 完成！")


if __name__ == "__main__":
    main()
