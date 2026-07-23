"""
对比测试 v33 和 v34，输出逐帧对比图
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

_KERNEL_CACHE = {}


# ========== v33 函数 ==========
def create_biased_kernel(kernel_size, bias_strength, device, dtype):
    pad = kernel_size // 2
    weights = torch.exp(torch.linspace(0, np.log(bias_strength), kernel_size, device=device))
    weights = weights / weights.mean()
    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    return kernel_2d


def detect_hole_edges(hole_mask):
    is_hole = hole_mask.unsqueeze(0).unsqueeze(0).float()
    kernel = torch.ones((1, 1, 3, 3), device=hole_mask.device, dtype=is_hole.dtype)
    neighbor_count = F.conv2d(1 - is_hole, kernel, padding=1)
    edge_mask = hole_mask & (neighbor_count[0, 0] > 0.01)
    return edge_mask


def fill_edge_with_nearest_bg_biased_v33(img, hole, edge_mask, near,
                                          kernel_size=5, bias_strength=5.0,
                                          bg_threshold=0.3):
    result = img.clone()
    filled_mask = torch.zeros_like(hole)
    device = hole.device
    bg_mask = ~hole & (near < bg_threshold)

    if not torch.any(edge_mask):
        return result, filled_mask

    edge_extended = edge_mask.clone()
    for _ in range(2):
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=1) \
                                     | torch.roll(edge_extended, shifts=-1, dims=1)
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=0) \
                                     | torch.roll(edge_extended, shifts=-1, dims=0)

    to_fill = edge_extended & hole

    if not torch.any(to_fill):
        return result, filled_mask

    pad = kernel_size // 2
    kernel1 = create_biased_kernel(kernel_size, bias_strength, device, img.dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)

    img_nchw = img.permute(2, 0, 1).unsqueeze(0)
    bg_mask_nchw = bg_mask.unsqueeze(0).unsqueeze(0).float()

    weighted_img = img_nchw * bg_mask_nchw
    rgb_sum = F.conv2d(weighted_img, kernel3, padding=pad, groups=3)
    weight_sum = F.conv2d(bg_mask_nchw, kernel1, padding=pad)

    avg = rgb_sum / weight_sum.clamp_min(1e-6)
    avg_hwc = avg[0].permute(1, 2, 0)

    valid_fill = to_fill & (weight_sum[0, 0] > 0.5)
    if torch.any(valid_fill):
        result[valid_fill] = avg_hwc[valid_fill]
        filled_mask[valid_fill] = True

    return result, filled_mask


def fast_inpaint_v33_biased(img, hole, near,
                             edge_kernel_size=5,
                             inner_kernel_size=11,
                             bias_strength=5.0,
                             bg_threshold=0.3,
                             max_iter=64):
    img = img.clone()
    hole = hole.clone()
    device = hole.device

    edge_mask = detect_hole_edges(hole)
    img, edge_filled = fill_edge_with_nearest_bg_biased_v33(
        img, hole, edge_mask, near, edge_kernel_size, bias_strength, bg_threshold
    )
    hole = hole & ~edge_filled

    if hole.sum().item() == 0:
        return img, 0, "边缘填补完成"

    cache_key = (inner_kernel_size, bias_strength, str(device), str(img.dtype).split('.')[-1])
    k1, k3 = _KERNEL_CACHE.get(cache_key, (None, None))

    if k1 is None:
        k1 = create_biased_kernel(inner_kernel_size, bias_strength, device, img.dtype)
        k3 = k1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[cache_key] = (k1, k3)

    pad = inner_kernel_size // 2
    prev_hole_count = hole.sum().item()
    no_progress_count = 0
    actual_iter = 0

    for it in range(max_iter):
        actual_iter += 1

        bg_weight = (~hole).unsqueeze(-1).float()
        weighted_img = img * bg_weight

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, k1, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole & (weight_sum[0, 0] > 0.5)

        if can_fill.sum().item() > 0:
            img[can_fill] = avg_hwc[can_fill]
            hole[can_fill] = False

        current_hole_count = hole.sum().item()
        if can_fill.sum().item() == 0 or current_hole_count >= prev_hole_count:
            no_progress_count += 1
            if no_progress_count >= 2:
                break
        else:
            no_progress_count = 0
        prev_hole_count = current_hole_count

        if not hole.any():
            break

    if hole.any():
        fallback_key = (15, bias_strength, str(device), str(img.dtype).split('.')[-1])
        k1_fallback, k3_fallback = _KERNEL_CACHE.get(fallback_key, (None, None))

        if k1_fallback is None:
            k1_fallback = create_biased_kernel(15, bias_strength, device, img.dtype)
            k3_fallback = k1_fallback.repeat(3, 1, 1, 1)
            _KERNEL_CACHE[fallback_key] = (k1_fallback, k3_fallback)

        pad_fallback = 7
        for _ in range(max_iter // 2):
            fillable_weight = (~hole).unsqueeze(-1).float()
            weighted_img = img * fillable_weight

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = fillable_weight.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3_fallback, padding=pad_fallback, groups=3)
            weight_sum = F.conv2d(weight_nchw, k1_fallback, padding=pad_fallback)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = hole & (weight_sum[0, 0] > 0.5)
            if can_fill.sum().item() > 0:
                img[can_fill] = avg_hwc[can_fill]
                hole[can_fill] = False
            else:
                break

            if not hole.any():
                break

    return img, actual_iter, "完成"


# ========== v34 函数 ==========
def create_strict_right_kernel(kernel_size, right_bias_decay, device, dtype):
    pad = kernel_size // 2
    distance_from_center = torch.arange(kernel_size, device=device) - pad
    left_mask = distance_from_center < 0
    right_distances = distance_from_center.float()
    weights = torch.exp(-right_distances * right_bias_decay)
    weights[left_mask] = 0.0
    if weights.sum() > 1e-6:
        weights = weights / weights.sum() * (pad + 1)
    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)
    return kernel_2d


def fill_edge_strict_right_v34(img, hole, edge_mask, near,
                               kernel_size=7, right_bias_decay=2.0,
                               bg_threshold=0.3):
    result = img.clone()
    filled_mask = torch.zeros_like(hole)
    device = hole.device
    bg_mask = ~hole & (near < bg_threshold)

    if not torch.any(edge_mask):
        return result, filled_mask

    edge_extended = edge_mask.clone()
    for _ in range(2):
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=-1, dims=1)  # 只向右
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=1, dims=0)
        edge_extended = edge_extended | torch.roll(edge_extended, shifts=-1, dims=0)

    to_fill = edge_extended & hole

    if not torch.any(to_fill):
        return result, filled_mask

    pad = kernel_size // 2
    kernel1 = create_strict_right_kernel(kernel_size, right_bias_decay, device, img.dtype)
    kernel3 = kernel1.repeat(3, 1, 1, 1)

    img_nchw = img.permute(2, 0, 1).unsqueeze(0)
    bg_mask_nchw = bg_mask.unsqueeze(0).unsqueeze(0).float()

    weighted_img = img_nchw * bg_mask_nchw
    rgb_sum = F.conv2d(weighted_img, kernel3, padding=pad, groups=3)
    weight_sum = F.conv2d(bg_mask_nchw, kernel1, padding=pad)

    avg = rgb_sum / weight_sum.clamp_min(1e-6)
    avg_hwc = avg[0].permute(1, 2, 0)

    valid_fill = to_fill & (weight_sum[0, 0] > 0.5)
    if torch.any(valid_fill):
        result[valid_fill] = avg_hwc[valid_fill]
        filled_mask[valid_fill] = True

    return result, filled_mask


def fast_inpaint_v34_strict_right(img, hole, near,
                                   edge_kernel_size=7,
                                   inner_kernel_size=15,
                                   right_bias_decay=2.0,
                                   bg_threshold=0.3,
                                   max_iter=64):
    img = img.clone()
    hole = hole.clone()
    device = hole.device

    edge_mask = detect_hole_edges(hole)
    img, edge_filled = fill_edge_strict_right_v34(
        img, hole, edge_mask, near, edge_kernel_size, right_bias_decay, bg_threshold
    )
    hole = hole & ~edge_filled

    if hole.sum().item() == 0:
        return img, 0, "边缘填补完成"

    cache_key = (inner_kernel_size, right_bias_decay, str(device), str(img.dtype).split('.')[-1])
    k1, k3 = _KERNEL_CACHE.get(cache_key, (None, None))

    if k1 is None:
        k1 = create_strict_right_kernel(inner_kernel_size, right_bias_decay, device, img.dtype)
        k3 = k1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[cache_key] = (k1, k3)

    pad = inner_kernel_size // 2
    prev_hole_count = hole.sum().item()
    no_progress_count = 0
    actual_iter = 0

    for it in range(max_iter):
        actual_iter += 1

        bg_weight = (~hole).unsqueeze(-1).float()
        weighted_img = img * bg_weight

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)

        rgb_sum = F.conv2d(img_nchw, k3, padding=pad, groups=3)
        weight_sum = F.conv2d(weight_nchw, k1, padding=pad)

        avg = rgb_sum / weight_sum.clamp_min(1e-6)
        avg_hwc = avg[0].permute(1, 2, 0)

        can_fill = hole & (weight_sum[0, 0] > 0.5)

        if can_fill.sum().item() > 0:
            img[can_fill] = avg_hwc[can_fill]
            hole[can_fill] = False

        current_hole_count = hole.sum().item()
        if can_fill.sum().item() == 0 or current_hole_count >= prev_hole_count:
            no_progress_count += 1
            if no_progress_count >= 2:
                break
        else:
            no_progress_count = 0
        prev_hole_count = current_hole_count

        if not hole.any():
            break

    if hole.any():
        fallback_key = (21, right_bias_decay, str(device), str(img.dtype).split('.')[-1])
        k1_fallback, k3_fallback = _KERNEL_CACHE.get(fallback_key, (None, None))

        if k1_fallback is None:
            k1_fallback = create_strict_right_kernel(21, right_bias_decay, device, img.dtype)
            k3_fallback = k1_fallback.repeat(3, 1, 1, 1)
            _KERNEL_CACHE[fallback_key] = (k1_fallback, k3_fallback)

        pad_fallback = 10
        for _ in range(max_iter // 2):
            fillable_weight = (~hole).unsqueeze(-1).float()
            weighted_img = img * fillable_weight

            img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
            weight_nchw = fillable_weight.permute(2, 0, 1).unsqueeze(0)

            rgb_sum = F.conv2d(img_nchw, k3_fallback, padding=pad_fallback, groups=3)
            weight_sum = F.conv2d(weight_nchw, k1_fallback, padding=pad_fallback)

            avg = rgb_sum / weight_sum.clamp_min(1e-6)
            avg_hwc = avg[0].permute(1, 2, 0)

            can_fill = hole & (weight_sum[0, 0] > 0.5)
            if can_fill.sum().item() > 0:
                img[can_fill] = avg_hwc[can_fill]
                hole[can_fill] = False
            else:
                break

            if not hole.any():
                break

    return img, actual_iter, "完成"


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


def edge_post_process_vectorized_v33(image, hole_mask, smooth_width=6, smooth_sigma=1.5):
    h, w = hole_mask.shape
    device = image.device
    result = image.clone()

    if smooth_width > 0:
        hole_int = hole_mask.long()
        edge_x = hole_int.argmax(dim=1)
        has_no_hole = (hole_int.sum(dim=1) == 0)
        edge_x[has_no_hole] = w

        k = 5
        pad = k // 2
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * smooth_sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + smooth_width) & hole_mask
        smooth_mask_3d = smooth_region.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    return result


def edge_post_process_vectorized_v34(image, hole_mask, smooth_width=6, smooth_sigma=1.5):
    h, w = hole_mask.shape
    device = image.device
    result = image.clone()

    if smooth_width > 0:
        hole_int = hole_mask.long()
        edge_x = hole_int.argmax(dim=1)
        has_no_hole = (hole_int.sum(dim=1) == 0)
        edge_x[has_no_hole] = w

        k = 5
        pad = k // 2
        gauss_kernel = torch.exp(-(torch.arange(k, device=device, dtype=torch.float32) - pad) ** 2 / (2 * smooth_sigma ** 2))
        gauss_kernel[:pad] = 0.0  # 左侧权重置 0
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = gauss_kernel.view(1, 1, k, 1).repeat(3, 1, 1, 1)

        img_4d = result.permute(2, 0, 1).unsqueeze(0)
        img_padded = F.pad(img_4d, (0, 0, pad, pad), mode='replicate')
        img_smoothed = F.conv2d(img_padded, gauss_kernel, groups=3)[0].permute(1, 2, 0)

        x_indices = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(h, w)
        edge_x_2d = edge_x.view(h, 1).expand(h, w)
        smooth_region = (x_indices >= edge_x_2d) & (x_indices < edge_x_2d + smooth_width) & hole_mask
        smooth_mask_3d = smooth_region.unsqueeze(-1)
        result = torch.where(smooth_mask_3d, img_smoothed, result)

    return result


def main():
    import time
    video_path = Path("/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4")
    output_dir_v33 = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v33_test/disp24_e5_i11_b10")
    output_dir_v34 = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v34_test/disp24_e7_i15_decay2")

    output_dir_v33.mkdir(parents=True, exist_ok=True)
    output_dir_v34.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location='cpu'))
    model = model.to(device).eval()
    print("模型加载完成")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"输入视频: {w_orig}×{h_orig}, {fps_orig:.1f} FPS, 共 {total_frames} 帧")
    print(f"只处理前 100 帧用于对比测试")

    input_size = 518
    scale = input_size / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = input_size
    else:
        depth_h = input_size

    dibr_h, dibr_w = h_orig, w_orig
    max_disparity = 24.0
    max_disparity_dibr = max_disparity * dibr_w / w_orig

    print(f"深度模型分辨率: {depth_w}×{depth_h}")
    print(f"DIBR 分辨率: {dibr_w}×{dibr_h}")

    frame_times_v33 = []
    frame_times_v34 = []

    max_frames = min(100, total_frames)

    for frame_idx in range(max_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        print(f"处理帧 {frame_idx + 1}/{max_frames}...", end='\r')

        # 深度推理（只做一次，两个版本共享）
        left_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(left_rgb_original).to(device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)
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
            size=(dibr_h, dibr_w),
            mode="bilinear", align_corners=False
        )[0, 0]
        disparity = near_score * max_disparity_dibr

        disparity_sharp, unreliable = b.sharpen_disparity_edges(
            disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
        )

        left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        disocclusion_band = project_disocclusion_bands_gpu(disparity_sharp, min_drop=3.5, min_band_width=8)
        hole_with_band = hole | disocclusion_band

        # ========== v33 处理 ==========
        t0 = time.time()
        right_with_band_v33 = right_warped.clone()
        right_with_band_v33[hole_with_band] = 0.0

        img_inpainted_v33, _, _ = fast_inpaint_v33_biased(
            right_with_band_v33, hole_with_band, near_score,
            edge_kernel_size=5, inner_kernel_size=11,
            bias_strength=10.0, bg_threshold=0.3, max_iter=64
        )

        final_right_v33 = edge_post_process_vectorized_v33(
            img_inpainted_v33, hole_with_band, smooth_width=6, smooth_sigma=1.5
        )
        t_v33 = time.time() - t0
        frame_times_v33.append(t_v33)

        # ========== v34 处理 ==========
        t0 = time.time()
        right_with_band_v34 = right_warped.clone()
        right_with_band_v34[hole_with_band] = 0.0

        img_inpainted_v34, _, _ = fast_inpaint_v34_strict_right(
            right_with_band_v34, hole_with_band, near_score,
            edge_kernel_size=7, inner_kernel_size=15,
            right_bias_decay=2.0, bg_threshold=0.3, max_iter=64
        )

        final_right_v34 = edge_post_process_vectorized_v34(
            img_inpainted_v34, hole_with_band, smooth_width=6, smooth_sigma=1.5
        )
        t_v34 = time.time() - t0
        frame_times_v34.append(t_v34)

        # 保存 SBS 对比图
        left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
        right_v33_uint8 = (final_right_v33 * 255).byte().cpu().numpy()
        right_v34_uint8 = (final_right_v34 * 255).byte().cpu().numpy()

        # v33: 左图 + v33 右图
        sbs_v33 = np.concatenate([left_uint8, right_v33_uint8], axis=1)
        cv2.imwrite(str(output_dir_v33 / f"frame_{frame_idx:04d}.png"), cv2.cvtColor(sbs_v33, cv2.COLOR_RGB2BGR))

        # v34: 左图 + v34 右图
        sbs_v34 = np.concatenate([left_uint8, right_v34_uint8], axis=1)
        cv2.imwrite(str(output_dir_v34 / f"frame_{frame_idx:04d}.png"), cv2.cvtColor(sbs_v34, cv2.COLOR_RGB2BGR))

        # 直接对比：v33 右图 + v34 右图
        comparison = np.concatenate([right_v33_uint8, right_v34_uint8], axis=1).copy()
        cv2.putText(comparison, "v33 (bias=10)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "v34 (strict right)", (w_orig + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(output_dir_v34.parent / f"comparison_{frame_idx:04d}.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    cap.release()

    print(f"\n\n✅ 处理完成！")
    print(f"\n⏱️  性能统计:")
    print(f"  v33 平均: {np.mean(frame_times_v33)*1000:.1f} ms/帧, 速度: {1/np.mean(frame_times_v33):.1f} FPS")
    print(f"  v34 平均: {np.mean(frame_times_v34)*1000:.1f} ms/帧, 速度: {1/np.mean(frame_times_v34):.1f} FPS")
    print(f"\n📁 输出目录:")
    print(f"  v33: {output_dir_v33}")
    print(f"  v34: {output_dir_v34}")
    print(f"  对比图: {output_dir_v34.parent}")


if __name__ == "__main__":
    main()
