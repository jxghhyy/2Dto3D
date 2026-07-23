"""
v45 两阶段混合填充：
  阶段 1 (5次)：严格右侧填充 → 填完 95%
  阶段 2 (3次)：垂直方向填充 → 快速搞定最后 5% 的右边界窄条

参数：dilate=4, kernel=15
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

DILATE_PIXELS = 4
KERNEL_SIZE = 15
STAGE1_ITERS = 5  # 严格右侧填充次数
STAGE2_ITERS = 3  # 垂直填充次数
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v45_two_stage")


def create_strict_right_kernel(kernel_size, device):
    """水平方向：只取右侧；垂直方向：自由取"""
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


def create_vertical_kernel(kernel_size, device):
    """纯垂直方向：上下自由取，水平方向只取中心"""
    # 构造一个垂直条形核：水平方向只有中心为1，上下全为1
    weights = torch.zeros(kernel_size, kernel_size, device=device, dtype=torch.float32)
    pad = kernel_size // 2
    weights[:, pad] = 1.0  # 中间一整列
    kernel_2d = weights.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d / kernel_2d.sum() * kernel_size  # 归一化
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
def inpaint_with_kernel(img, hole, kernel, k3, max_iter):
    """用指定卷积核填充"""
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v45 两阶段] 使用设备: {device}")
    print(f"[v45 两阶段] GPU 型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"[v45 两阶段] 参数: dilate={DILATE_PIXELS}, kernel={KERNEL_SIZE}")
    print(f"[v45 两阶段] 阶段 1: {STAGE1_ITERS} 次严格右侧填充")
    print(f"[v45 两阶段] 阶段 2: {STAGE2_ITERS} 次垂直填充")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # 预创建两个卷积核
    kernel_right = create_strict_right_kernel(KERNEL_SIZE, device)
    k3_right = kernel_right.repeat(3, 1, 1, 1)

    kernel_vert = create_vertical_kernel(KERNEL_SIZE, device)
    k3_vert = kernel_vert.repeat(3, 1, 1, 1)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[v45 两阶段] 视频: {fps:.1f} fps, {total_frames} 帧, {w_orig}×{h_orig}")

    # 输出视频
    out_path = OUT_DIR / f"v45_dilate{DILATE_PIXELS}_kernel{KERNEL_SIZE}_2stage.mp4"
    sbs_out_path = OUT_DIR / f"v45_dilate{DILATE_PIXELS}_kernel{KERNEL_SIZE}_2stage_sbs.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))
    sbs_writer = cv2.VideoWriter(str(sbs_out_path), fourcc, fps, (w_orig * 2, h_orig))

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

    frame_times = []
    remaining_after_stage1 = []
    remaining_final = []

    pbar = tqdm(range(total_frames), desc="处理帧")
    for frame_idx in pbar:
        t0 = time.time()

        ok, frame_bgr = cap.read()
        if not ok:
            break

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
        hole_dilated = dilate_hole_right_fixed(hole, DILATE_PIXELS, device)
        right_with_hole = right_warped.clone()
        right_with_hole[hole_dilated] = 0.0

        # ========== 两阶段填充 ==========
        # 阶段 1：严格右侧填充
        result, hole_after_s1 = inpaint_with_kernel(
            right_with_hole, hole_dilated, kernel_right, k3_right, STAGE1_ITERS
        )
        remaining_after_stage1.append(hole_after_s1.sum().item())

        # 阶段 2：垂直方向填充
        result, hole_final = inpaint_with_kernel(
            result, hole_after_s1, kernel_vert, k3_vert, STAGE2_ITERS
        )
        remaining_final.append(hole_final.sum().item())

        # 写视频
        result_np = (result.cpu().numpy() * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        out_writer.write(result_bgr)

        # SBS
        left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
        left_bgr = cv2.cvtColor(left_uint8, cv2.COLOR_RGB2BGR)
        sbs = np.concatenate([left_bgr, result_bgr], axis=1)
        sbs_writer.write(sbs)

        t1 = time.time()
        frame_times.append(t1 - t0)
        pbar.set_postfix({
            "t/frame": f"{np.mean(frame_times[-10:])*1000:.0f}ms",
            "rem": f"{remaining_final[-1]}"
        })

    cap.release()
    out_writer.release()
    sbs_writer.release()

    print(f"\n[v45 两阶段] ✅ 完成！")
    print(f"  共处理 {len(frame_times)} 帧")
    print(f"  平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    print(f"  阶段 1 后平均剩余: {np.mean(remaining_after_stage1):.0f} 像素")
    print(f"  最终平均剩余: {np.mean(remaining_final):.0f} 像素")
    print(f"  输出视频: {out_path}")
    print(f"  SBS 对比: {sbs_out_path}")


if __name__ == "__main__":
    main()
