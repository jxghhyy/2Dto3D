"""
v52: 优先从右向左填充的新思路

核心逻辑：
1. 默认：纯从右向左的不对称卷积（右侧全宽，左侧 0）
   - 背景永远在空洞右侧，所以直接从右向左扩散
2. 例外：只有空洞的最右边缘（右侧无有效像素），才用从左向右的卷积

测试：不同卷积核尺寸 (31, 21, 15, 11, 7) 的效果对比
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"

# 测试的卷积核尺寸
KERNEL_SIZES = [31, 21, 15, 11, 7]


def create_right_only_kernel(kernel_size, device):
    """
    创建纯从右向左的卷积核
    左侧全部为 0，右侧全部为 1
    这样只能从右侧（背景）获取像素填充
    """
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    # 水平方向：从 0 到 half（中心）都为 1
    # 这样当卷积时，中心像素只能看到自己右侧的内容
    kernel[0, 0, :, :half + 1] = 1.0
    return kernel


def create_left_only_kernel(kernel_size, device):
    """
    创建纯从左向右的卷积核（用于最右边缘的 fallback）
    右侧全部为 0，左侧全部为 1
    """
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    # 水平方向：从 half 到 end 都为 1
    # 这样当卷积时，中心像素只能看到自己左侧的内容
    kernel[0, 0, :, half:] = 1.0
    return kernel


@torch.no_grad()
def dilate_right_gpu(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    """只向右膨胀（v51 修复版）"""
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


@torch.no_grad()
def find_rightmost_hole(hole, kernel_size):
    """
    找到空洞的最右边缘（右侧没有有效像素的空洞像素）
    这些像素无法从右侧获取信息，需要 fallback 到从左向右
    """
    h, w = hole.shape
    half = kernel_size // 2

    # 方法：对于每个空洞像素，检查其右侧 half 范围内是否有非空洞像素
    # 如果没有，则属于最右边缘

    # 从右向左扫描，找到每行空洞的最右位置
    rightmost_mask = torch.zeros_like(hole)

    for y in range(h):
        row = hole[y]
        if not row.any():
            continue
        # 找到这一行所有空洞像素的 x 坐标
        hole_x = torch.where(row)[0]
        max_x = hole_x.max().item()
        # 从 max_x 向左，直到碰到非空洞像素 或者 超出 kernel 范围
        for x in range(max_x, max(-1, max_x - half * 2), -1):
            if x < 0 or not hole[y, x]:
                break
            # 检查右侧是否有有效像素
            right_has_pixel = False
            for rx in range(x + 1, min(w, x + half + 1)):
                if not hole[y, rx]:
                    right_has_pixel = True
                    break
            if not right_has_pixel:
                rightmost_mask[y, x] = True

    return rightmost_mask


@torch.no_grad()
def inpaint_right_first(img, hole, kernel_right, kernel_left, max_iter=8):
    """
    优先从右向左填充
    1. 默认：用 kernel_right（从右向左）
    2. 只有右侧没有像素的空洞，用 kernel_left（从左向右）
    """
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_right.shape[-1] // 2
    kernel_size = kernel_right.shape[-1]

    k3_right = kernel_right.repeat(3, 1, 1, 1)
    k3_left = kernel_left.repeat(3, 1, 1, 1)

    for it in range(max_iter):
        valid_mask = (~hole_cur).unsqueeze(-1).float()
        weighted_img = result * valid_mask

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = valid_mask.permute(2, 0, 1).unsqueeze(0)

        # 1. 从右向左的卷积
        rgb_sum_right = F.conv2d(img_nchw, k3_right, padding=pad, groups=3)
        weight_sum_right = F.conv2d(weight_nchw, kernel_right, padding=pad)

        # 2. 从左向右的卷积（fallback）
        rgb_sum_left = F.conv2d(img_nchw, k3_left, padding=pad, groups=3)
        weight_sum_left = F.conv2d(weight_nchw, kernel_left, padding=pad)

        # 找到无法从右侧填充的像素（最右边缘）
        rightmost_mask = find_rightmost_hole(hole_cur, kernel_size)

        # 合并：右侧有像素用右卷积，右侧没像素用左卷积
        # 注意：这里简化为直接用右卷积，能填充的就填充，不能填充的下一轮迭代继续
        # 只有当右侧完全无法填充时，才用左卷积填充最右边缘
        can_fill_right = hole_cur & (weight_sum_right[0, 0] > 0.5)
        can_fill_left = hole_cur & (weight_sum_right[0, 0] < 0.5) & (weight_sum_left[0, 0] > 0.5)

        can_fill = can_fill_right | can_fill_left
        if can_fill.sum().item() == 0:
            break

        # 计算平均值
        avg_right = rgb_sum_right / weight_sum_right.clamp_min(1e-6)
        avg_left = rgb_sum_left / weight_sum_left.clamp_min(1e-6)
        avg_hwc_right = avg_right[0].permute(1, 2, 0)
        avg_hwc_left = avg_left[0].permute(1, 2, 0)

        # 合并结果
        avg_hwc = torch.where(
            can_fill_right.unsqueeze(-1),
            avg_hwc_right,
            avg_hwc_left
        )

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        if not hole_cur.any():
            break

    return result, hole_cur


def process_with_kernel_size(kernel_size, left_rgb_tensor, right_warped, hole_dilated, device):
    """用指定卷积核尺寸处理单帧"""
    kernel_right = create_right_only_kernel(kernel_size, device)
    kernel_left = create_left_only_kernel(kernel_size, device)

    result, hole_final = inpaint_right_first(
        right_warped, hole_dilated, kernel_right, kernel_left, 8
    )

    remaining = hole_final.sum().item()
    return result, remaining


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v52 优先从右向左] 设备: {device}")
    print(f"[v52 优先从右向左] 测试卷积核尺寸: {KERNEL_SIZES}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    h_orig, w_orig = frame_bgr.shape[:2]
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

    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    img_resized = F.interpolate(img, size=(depth_h, depth_w), mode="bilinear", align_corners=False)
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
    disparity = near_score * 24.0

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    hole_dilated = dilate_right_gpu(right_warped, hole, 8, 0.15, device)
    print(f"原始空洞: {hole.sum().item():,}, 膨胀后: {hole_dilated.sum().item():,}")

    OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v52_kernel_size_test")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 测试不同卷积核尺寸
    results = []
    for ks in KERNEL_SIZES:
        print(f"\n测试卷积核 {ks}x{ks}...")
        t0 = time.time()
        result, remaining = process_with_kernel_size(ks, left_rgb_tensor, right_warped, hole_dilated, device)
        t1 = time.time()
        results.append((ks, remaining, t1 - t0))
        print(f"  剩余空洞: {remaining}, 耗时: {(t1-t0)*1000:.1f}ms")

        # 保存结果
        result_np = (result.cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(OUT_DIR / f"result_k{ks}.png"), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

        # SBS 对比
        left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
        sbs = np.concatenate([left_uint8, result_np], axis=1)
        cv2.imwrite(str(OUT_DIR / f"sbs_k{ks}.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

        # 裁剪关键区域
        for crop_name, (y1, y2, x1, x2) in [
            ("face", (200, 500, 600, 900)),
            ("shoulder", (550, 750, 750, 1050)),
        ]:
            crop = result_np[y1:y2, x1:x2]
            cv2.imwrite(str(OUT_DIR / f"crop_{crop_name}_k{ks}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 50)
    print("【不同卷积核尺寸对比】")
    print(f"{'尺寸':<10} {'剩余空洞':<12} {'耗时(ms)':<10}")
    print("-" * 50)
    for ks, remaining, t in results:
        print(f"{ks:<10} {remaining:<12,} {t*1000:<10.1f}")
    print("=" * 50)
    print(f"\n输出目录: {OUT_DIR}")
    print(f"重点看:")
    print(f"  - crop_shoulder_k*.png → 肩膀颜色渗色情况")
    print(f"  - crop_face_k*.png → 脸部边缘效果")


if __name__ == "__main__":
    main()
