"""
v56最终版：正确的自适应核策略

关键修正：
1. 阶段1（右→左）：每个核大小迭代多次，直到真填不动了再减核
   - 核=11：迭代到填不动 → 核=9：迭代到填不动 → ... → 核=3
2. 阶段2（左→右）：只填最右侧没有背景像素的区域，不是全部剩余
   - 检测标准：空洞右边界 == 图像右边界（右侧没像素）
   - 这些区域可以从左向右填，不设上限
   - 其他区域不填（防止前景颜色泄漏）
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
INITIAL_KERNEL = 11
MIN_KERNEL = 3
MAX_ITER_PER_KERNEL = 10  # 每个核最多迭代10次


def create_directional_kernel(kernel_size, direction, device):
    """
    创建方向卷积核
    direction: 'right_to_left' = 看右侧，向左填充
               'left_to_right' = 看左侧，向右填充
    """
    half = kernel_size // 2
    kernel = torch.zeros((kernel_size, kernel_size), device=device, dtype=torch.float32)

    if direction == 'right_to_left':
        # 从右向左填充：中心 + 右侧像素
        kernel[:, half:] = 1.0
    elif direction == 'left_to_right':
        # 从左向右填充：左侧 + 中心像素
        kernel[:, :half + 1] = 1.0

    kernel = kernel / kernel.sum()
    kernel_2d = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel_2d


@torch.no_grad()
def find_rightmost_only_region(hole, img_width, device):
    """
    检测"右侧完全没有像素"的区域
    返回：mask，只包含那些右边界贴着图像右边界的空洞
    """
    hole_np = hole.cpu().numpy().astype(np.uint8)
    h, w = hole_np.shape

    # 每行找最右侧的空洞x坐标
    rightmost_x = np.full(h, -1, dtype=np.int32)
    for y in range(h):
        row = hole_np[y]
        hole_x = np.where(row > 0)[0]
        if len(hole_x) > 0:
            rightmost_x[y] = hole_x[-1]

    # 判断：最右侧空洞是否在图像右边界附近
    # 如果右边界 >= w - 3，说明右侧没有有效像素
    is_rightmost = (rightmost_x >= w - 3)

    # 生成mask：只保留这些行的空洞
    rightmost_mask = np.zeros_like(hole_np, dtype=bool)
    for y in range(h):
        if is_rightmost[y]:
            rightmost_mask[y] = hole_np[y] > 0

    print(f"    检测到最右侧无像素区域: {rightmost_mask.sum()} 像素")
    return torch.from_numpy(rightmost_mask).to(device)


@torch.no_grad()
def fill_hole_final(img, hole, device):
    """
    最终版两阶段填充：
    1. 从右向左填充，每个核迭代到填不动再减核
    2. 只对"最右侧无像素"的区域从左向右填充
    """
    result = img.clone()
    hole_cur = hole.clone()
    h, w = hole.shape

    # ========== 阶段1: 从右向左填充，每个核迭代到填不动 ==========
    print(f"\n  阶段1: 从右向左填充（核 {INITIAL_KERNEL} → {MIN_KERNEL}，每个核迭代到填不动）")

    kernel_size = INITIAL_KERNEL
    total_iter = 0

    while kernel_size >= MIN_KERNEL:
        kernel = create_directional_kernel(kernel_size, 'right_to_left', device)
        k3 = kernel.repeat(3, 1, 1, 1)
        pad = kernel_size // 2

        iter_for_this_kernel = 0
        while iter_for_this_kernel < MAX_ITER_PER_KERNEL:
            valid_mask = (~hole_cur).unsqueeze(-1).float()
            weighted_img = result * valid_mask

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
            weight_sum = F.conv2d(weight_nchw, kernel, padding=pad)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = hole_cur & (weight_sum[0, 0] > 0.5)
            filled = can_fill.sum().item()

            if filled == 0:
                break

            result[can_fill] = avg_hwc[can_fill]
            hole_cur[can_fill] = False
            iter_for_this_kernel += 1
            total_iter += 1

        print(f"    核={kernel_size}, 迭代{iter_for_this_kernel}次, 剩余={hole_cur.sum().item()}")

        # 核减小
        kernel_size -= 2

    print(f"  阶段1完成: 共{total_iter}轮迭代")

    # ========== 阶段2: 只对最右侧无像素区域从左向右填充 ==========
    remaining = hole_cur.sum().item()
    print(f"\n  阶段2: 剩余 {remaining} 像素，只填充最右侧无像素区域")

    # 检测最右侧区域
    rightmost_mask = find_rightmost_only_region(hole_cur, w, device)

    if rightmost_mask.sum() > 0:
        # 只填充这个区域
        region_hole = hole_cur & rightmost_mask

        kernel = create_directional_kernel(3, 'left_to_right', device)
        k3 = kernel.repeat(3, 1, 1, 1)

        stage2_iter = 0
        while stage2_iter < 1000:  # 设置上限防止死循环
            valid_mask = (~hole_cur).unsqueeze(-1).float()
            weighted_img = result * valid_mask

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3, padding=1, groups=3)
            weight_sum = F.conv2d(weight_nchw, kernel, padding=1)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = region_hole & hole_cur & (weight_sum[0, 0] > 0.5)
            filled = can_fill.sum().item()

            if filled == 0:
                break

            result[can_fill] = avg_hwc[can_fill]
            hole_cur[can_fill] = False
            region_hole[can_fill] = False
            stage2_iter += 1

            if stage2_iter % 50 == 0:
                print(f"    已迭代{stage2_iter}次, 区域剩余={region_hole.sum().item()}")

        print(f"  阶段2完成: 共{stage2_iter}轮迭代, 该区域剩余={region_hole.sum().item()}")
    else:
        print(f"  阶段2跳过: 没有最右侧无像素区域")

    print(f"\n  最终剩余空洞: {hole_cur.sum().item()}")

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

    # v56 最终版填充
    result, hole_final = fill_hole_final(right_with_hole, hole_dilated, device)

    result_np = (result.cpu().numpy() * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

    # 剩余空洞可视化
    remaining_viz = result_bgr.copy()
    remaining_mask = hole_final.cpu().numpy()
    remaining_viz[remaining_mask] = [0, 0, 255]  # 红色标记剩余空洞

    # 拼接对比图
    original_right = (right_warped.cpu().numpy() * 255).astype(np.uint8)
    original_right_bgr = cv2.cvtColor(original_right, cv2.COLOR_RGB2BGR)

    # 四图拼接：左图 | 原图右视图 | 填充结果 | 剩余空洞（红色标记）
    combined = np.hstack([frame_bgr, original_right_bgr, result_bgr, remaining_viz])

    cv2.imwrite(str(OUT_DIR / f"{frame_name}_final_result.png"), result_bgr)
    cv2.imwrite(str(OUT_DIR / f"{frame_name}_final_combined.png"), combined)

    print(f"\n结果已保存: {frame_name}_final_result.png / {frame_name}_final_combined.png")
    return result_bgr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v56最终版] 设备: {device}")
    print(f"[v56最终版] 初始核={INITIAL_KERNEL}, 最小核={MIN_KERNEL}")
    print(f"[v56最终版] 输出目录: {OUT_DIR}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

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
    print("\n[v56最终版] 完成！")


if __name__ == "__main__":
    main()
