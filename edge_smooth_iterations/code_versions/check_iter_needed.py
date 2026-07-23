"""
检查 dilate=4 + kernel=15 实际需要多少次迭代才能填满
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
    pad = kernel_size // 2
    x_indices = torch.arange(kernel_size, device=device) - pad
    horizontal_mask = x_indices >= 0
    horizontal_weights = torch.ones(kernel_size, device=device, dtype=torch.float32)
    horizontal_weights[~horizontal_mask] = 0.0
    kernel_1d = horizontal_weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    num_nonzero = horizontal_weights.sum().item() * kernel_size
    if num_nonzero > 0:
        kernel_2d = kernel_2d / kernel_2d.sum() * num_nonzero
    return kernel_2d


@torch.no_grad()
def dilate_hole_right_fixed(hole, dilate_pixels, device):
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()

    for y in range(h):
        row = hole_np[y]
        if not row.any():
            continue
        indices = np.where(row)[0]
        regions = []
        if len(indices) > 0:
            start_x = indices[0]
            prev_x = indices[0]
            for i in range(1, len(indices)):
                x = indices[i]
                if x > prev_x + 1:
                    regions.append((start_x, prev_x))
                    start_x = x
                prev_x = x
            regions.append((start_x, prev_x))
        for (start_x, end_x) in regions:
            new_end_x = min(w - 1, end_x + dilate_pixels)
            hole_dilated[y, end_x:new_end_x + 1] = True

    return torch.from_numpy(hole_dilated).to(device)


@torch.no_grad()
def inpaint_strict_right_count_iter(img, hole, kernel, k3, max_iter=50):
    """统计实际需要多少次迭代"""
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2
    device = hole.device

    actual_iters = 0
    remaining_per_iter = []

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
        remaining = hole_cur.sum().item()
        remaining_per_iter.append(remaining)

        actual_iters += 1

        if filled_count == 0:
            break

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        if not hole_cur.any():
            break

    return actual_iters, remaining_per_iter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
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

    # 深度推理参数
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
    max_disparity = 24.0 * w_orig / w_orig

    # 深度推理
    left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)

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
    disparity = near_score * max_disparity

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    # 向右膨胀
    hole_dilated = dilate_hole_right_fixed(hole, 4, device)
    right_with_hole = right_warped.clone()
    right_with_hole[hole_dilated] = 0.0

    print(f"空洞总数: {hole_dilated.sum().item():,} 像素")
    print(f"测试 kernel=15, dilate=4...")

    kernel = create_strict_right_kernel(15, device)
    k3 = kernel.repeat(3, 1, 1, 1)

    actual_iters, remaining_per_iter = inpaint_strict_right_count_iter(
        right_with_hole, hole_dilated, kernel, k3, max_iter=100
    )

    print(f"\n实际迭代次数: {actual_iters}")
    print(f"\n每次迭代剩余空洞:")
    for i, rem in enumerate(remaining_per_iter[:15]):
        print(f"  迭代 {i+1:2d}: {rem:,} 像素")

    if len(remaining_per_iter) > 15:
        print(f"  ... 之后 {len(remaining_per_iter) - 15} 次迭代继续减少 ...")

    print(f"\n最终剩余空洞: {remaining_per_iter[-1]:,} 像素")
    print(f"\n结论: max_iter 设为 {actual_iters} 就够了！")


if __name__ == "__main__":
    main()
