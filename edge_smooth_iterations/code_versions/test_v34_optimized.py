"""
v34 优化版：使用较小衰减因子 + 空洞卷积加速
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


def create_strict_right_kernel(kernel_size, right_bias_decay, device, dtype):
    """
    创建严格的右侧-only卷积核
    """
    pad = kernel_size // 2
    distance_from_center = torch.arange(kernel_size, device=device) - pad
    left_mask = distance_from_center < 0
    right_distances = distance_from_center.float()

    # 较小的衰减因子 = 更均匀的右侧权重 = 更快的填充速度
    weights = torch.exp(-right_distances * right_bias_decay)
    weights[left_mask] = 0.0

    if weights.sum() > 1e-6:
        weights = weights / weights.sum() * (pad + 1)

    kernel_1d = weights.view(1, 1, 1, kernel_size)
    kernel_2d = kernel_1d.repeat(1, 1, kernel_size, 1)

    return kernel_2d


def fast_inpaint_v34_optimized(img, hole, near,
                               inner_kernel_size=11,
                               right_bias_decay=0.5,  # 更小的衰减因子
                               dilation=2,  # 空洞卷积加速
                               bg_threshold=0.3,
                               max_iter=32):
    """
    v34 优化版：
    - 更小的衰减因子，让右侧更远的像素也能参与计算
    - 空洞卷积，不增加计算量的情况下扩大感受野
    """
    img = img.clone()
    hole = hole.clone()
    device = hole.device

    pad = inner_kernel_size // 2
    actual_pad = pad * dilation  # 考虑空洞膨胀

    cache_key = (inner_kernel_size, right_bias_decay, dilation, str(device), str(img.dtype).split('.')[-1])
    k1, k3 = _KERNEL_CACHE.get(cache_key, (None, None))

    if k1 is None:
        k1 = create_strict_right_kernel(inner_kernel_size, right_bias_decay, device, img.dtype)
        k3 = k1.repeat(3, 1, 1, 1)
        _KERNEL_CACHE[cache_key] = (k1, k3)

    prev_hole_count = hole.sum().item()
    no_progress_count = 0
    actual_iter = 0

    for it in range(max_iter):
        actual_iter += 1

        bg_weight = (~hole).unsqueeze(-1).float()
        weighted_img = img * bg_weight

        img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
        weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)

        # 使用空洞卷积加速填充
        rgb_sum = F.conv2d(img_nchw, k3, padding=actual_pad, dilation=dilation, groups=3)
        weight_sum = F.conv2d(weight_nchw, k1, padding=actual_pad, dilation=dilation)

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

    return img, actual_iter


def main():
    import time
    video_path = Path("/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4")
    output_dir_v34_orig = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v34_test/original")
    output_dir_v34_opt = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v34_test/optimized")
    output_dir_v33 = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v33_test/reference")

    for d in [output_dir_v34_orig, output_dir_v34_opt, output_dir_v33]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_size = 518
    scale = input_size / max(h_orig, w_orig)
    depth_h = max(14, int(round(h_orig * scale / 14)) * 14)
    depth_w = max(14, int(round(w_orig * scale / 14)) * 14)
    if w_orig >= h_orig:
        depth_w = input_size
    else:
        depth_h = input_size

    dibr_h, dibr_w = h_orig, w_orig
    max_disparity = 24.0 * dibr_w / w_orig

    times_v33 = []
    times_v34_orig = []
    times_v34_opt = []
    iters_v33 = []
    iters_v34_orig = []
    iters_v34_opt = []

    max_frames = min(10, total_frames)

    for frame_idx in range(max_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        print(f"处理帧 {frame_idx + 1}/{max_frames}...", end="\r")

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
        disparity = near_score * max_disparity

        disparity_sharp, unreliable = b.sharpen_disparity_edges(
            disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
        )

        left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
        right_warped, hole = b.forward_warp_excluding_source(
            left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
        )

        right_with_band = right_warped.clone()
        right_with_band[hole] = 0.0

        # v33 (参考)
        def create_v33_kernel(kernel_size, bias_strength):
            pad = kernel_size // 2
            weights = torch.exp(torch.linspace(0, np.log(bias_strength), kernel_size, device=device))
            weights = weights / weights.mean()
            kernel_1d = weights.view(1, 1, 1, kernel_size)
            return kernel_1d.repeat(1, 1, kernel_size, 1)

        k1_v33 = create_v33_kernel(11, 10.0)
        k3_v33 = k1_v33.repeat(3, 1, 1, 1)

        def inpaint_v33(img, hole):
            img = img.clone()
            hole = hole.clone()
            pad = 5
            iters = 0
            for _ in range(64):
                iters += 1
                bg_weight = (~hole).unsqueeze(-1).float()
                weighted_img = img * bg_weight
                img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
                weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)
                rgb_sum = F.conv2d(img_nchw, k3_v33, padding=pad, groups=3)
                weight_sum = F.conv2d(weight_nchw, k1_v33, padding=pad)
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
            return img, iters

        # v34 原版
        def create_v34_orig_kernel(kernel_size, decay):
            pad = kernel_size // 2
            distance_from_center = torch.arange(kernel_size, device=device) - pad
            left_mask = distance_from_center < 0
            right_distances = distance_from_center.float()
            weights = torch.exp(-right_distances * decay)
            weights[left_mask] = 0.0
            weights = weights / weights.sum() * (pad + 1)
            kernel_1d = weights.view(1, 1, 1, kernel_size)
            return kernel_1d.repeat(1, 1, kernel_size, 1)

        k1_v34_orig = create_v34_orig_kernel(15, 2.0)
        k3_v34_orig = k1_v34_orig.repeat(3, 1, 1, 1)

        def inpaint_v34_orig(img, hole):
            img = img.clone()
            hole = hole.clone()
            pad = 7
            iters = 0
            for _ in range(64):
                iters += 1
                bg_weight = (~hole).unsqueeze(-1).float()
                weighted_img = img * bg_weight
                img_nchw = weighted_img.permute(2, 0, 1).unsqueeze(0)
                weight_nchw = bg_weight.permute(2, 0, 1).unsqueeze(0)
                rgb_sum = F.conv2d(img_nchw, k3_v34_orig, padding=pad, groups=3)
                weight_sum = F.conv2d(weight_nchw, k1_v34_orig, padding=pad)
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
            return img, iters

        torch.cuda.synchronize()
        t0 = time.time()
        result_v33, iter_v33 = inpaint_v33(right_with_band, hole)
        torch.cuda.synchronize()
        t_v33 = time.time() - t0
        times_v33.append(t_v33)
        iters_v33.append(iter_v33)

        torch.cuda.synchronize()
        t0 = time.time()
        result_v34_orig, iter_v34_orig = inpaint_v34_orig(right_with_band, hole)
        torch.cuda.synchronize()
        t_v34_orig = time.time() - t0
        times_v34_orig.append(t_v34_orig)
        iters_v34_orig.append(iter_v34_orig)

        torch.cuda.synchronize()
        t0 = time.time()
        result_v34_opt, iter_v34_opt = fast_inpaint_v34_optimized(
            right_with_band, hole, near_score,
            inner_kernel_size=11, right_bias_decay=0.5, dilation=2
        )
        torch.cuda.synchronize()
        t_v34_opt = time.time() - t0
        times_v34_opt.append(t_v34_opt)
        iters_v34_opt.append(iter_v34_opt)

        # 保存对比图
        left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
        v33_uint8 = (result_v33 * 255).byte().cpu().numpy()
        v34_orig_uint8 = (result_v34_orig * 255).byte().cpu().numpy()
        v34_opt_uint8 = (result_v34_opt * 255).byte().cpu().numpy()

        comparison = np.concatenate([v33_uint8, v34_orig_uint8, v34_opt_uint8], axis=1)
        cv2.putText(comparison, f"v33 ({t_v33*1000:.0f}ms, iter={iter_v33})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"v34 orig ({t_v34_orig*1000:.0f}ms, iter={iter_v34_orig})", (w_orig + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"v34 opt ({t_v34_opt*1000:.0f}ms, iter={iter_v34_opt})", (w_orig*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(output_dir_v34_opt.parent / f"triple_{frame_idx:03d}.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    cap.release()

    print(f"\n{'=' * 70}")
    print(f"性能对比（{max_frames} 帧平均）：")
    print(f"  v33:           {np.mean(times_v33)*1000:6.1f} ms, 平均迭代 {np.mean(iters_v33):.1f} 次")
    print(f"  v34 original:  {np.mean(times_v34_orig)*1000:6.1f} ms, 平均迭代 {np.mean(iters_v34_orig):.1f} 次")
    print(f"  v34 optimized: {np.mean(times_v34_opt)*1000:6.1f} ms, 平均迭代 {np.mean(iters_v34_opt):.1f} 次")
    print(f"\n  v34 opt 相对于 v33: {np.mean(times_v34_opt)/np.mean(times_v33):.2f}x")
    print(f"  v34 opt 相对于 orig: {np.mean(times_v34_orig)/np.mean(times_v34_opt):.2f}x 加速")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
