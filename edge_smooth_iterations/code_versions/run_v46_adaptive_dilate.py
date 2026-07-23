"""
v46 自适应向右膨胀 + 三阶段填充：

核心改进：自适应膨胀量
  - 对每个空洞区域，测量其宽度（横向长度）
  - 宽度 ≈ max_disparity（24像素） → 最大膨胀 5 像素
  - 宽度越小 → 膨胀量按比例线性缩小
  - 避免小洞过度膨胀造成黑边

三阶段填充：
  1. 5次严格右侧填充
  2. 3次垂直填充
  3. 2次全向小核清理（最终扫尾）
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

# 参数
MAX_DISPARITY = 24.0
MAX_DILATE = 5  # 最大膨胀量（对应宽度=24的空洞）
KERNEL_SIZE = 15
STAGE1_ITERS = 5  # 严格右侧
STAGE2_ITERS = 3  # 垂直
STAGE3_ITERS = 2  # 全向清理
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v46_adaptive_dilate")


def create_strict_right_kernel(kernel_size, device):
    """严格右侧核"""
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
    """纯垂直核"""
    weights = torch.zeros(kernel_size, kernel_size, device=device, dtype=torch.float32)
    pad = kernel_size // 2
    weights[:, pad] = 1.0
    kernel_2d = weights.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d / kernel_2d.sum() * kernel_size
    return kernel_2d


def create_full_kernel(kernel_size, device):
    """全向均匀核（最终清理用）"""
    weights = torch.ones(kernel_size, kernel_size, device=device, dtype=torch.float32)
    kernel_2d = weights.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d / kernel_2d.sum() * kernel_size * kernel_size
    return kernel_2d


@torch.no_grad()
def dilate_hole_right_adaptive(hole, max_dilate, max_disparity, device):
    """
    ✅ 自适应向右膨胀：
    对每行中的每个连续空洞区域：
      1. 测量空洞宽度 = end_x - start_x + 1
      2. 膨胀像素 = clamp(宽度 / max_disparity * max_dilate, 1, max_dilate)
      3. 只向右膨胀
    """
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()

    total_original = hole_np.sum()
    total_added = 0

    for y in range(h):
        row = hole_np[y]
        if not row.any():
            continue

        # 找所有连续空洞区域
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

        # 对每个区域，根据宽度自适应膨胀
        for (start_x, end_x) in regions:
            width = end_x - start_x + 1

            # 自适应膨胀量：线性映射，最小1，最大max_dilate
            dilate_pixels = int(round(width / max_disparity * max_dilate))
            dilate_pixels = max(1, min(max_dilate, dilate_pixels))

            new_end_x = min(w - 1, end_x + dilate_pixels)
            hole_dilated[y, end_x:new_end_x + 1] = True

    total_dilated = hole_dilated.sum()
    total_added = total_dilated - total_original

    return torch.from_numpy(hole_dilated).to(device), total_added


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
    print(f"[v46 自适应膨胀] 使用设备: {device}")
    print(f"[v46 自适应膨胀] 参数: max_dilate={MAX_DILATE}, max_disparity={MAX_DISPARITY}")
    print(f"[v46 自适应膨胀] 核大小: {KERNEL_SIZE}")
    print(f"[v46 自适应膨胀] 阶段 1: {STAGE1_ITERS} 次严格右侧")
    print(f"[v46 自适应膨胀] 阶段 2: {STAGE2_ITERS} 次垂直")
    print(f"[v46 自适应膨胀] 阶段 3: {STAGE3_ITERS} 次全向清理")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # 预创建三个卷积核
    kernel_right = create_strict_right_kernel(KERNEL_SIZE, device)
    k3_right = kernel_right.repeat(3, 1, 1, 1)

    kernel_vert = create_vertical_kernel(KERNEL_SIZE, device)
    k3_vert = kernel_vert.repeat(3, 1, 1, 1)

    kernel_full = create_full_kernel(7, device)  # 清理用小核
    k3_full = kernel_full.repeat(3, 1, 1, 1)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[v46 自适应膨胀] 视频: {fps:.1f} fps, {total_frames} 帧, {w_orig}×{h_orig}")

    # 输出视频
    out_path = OUT_DIR / f"v46_adaptive_dilate{MAX_DILATE}_kernel{KERNEL_SIZE}.mp4"
    sbs_out_path = OUT_DIR / f"v46_adaptive_dilate{MAX_DILATE}_kernel{KERNEL_SIZE}_sbs.mp4"

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
    max_disparity = MAX_DISPARITY * w_orig / w_orig

    frame_times = []
    added_pixels_list = []
    remaining_after_s1 = []
    remaining_after_s2 = []
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

        # ========== 自适应向右膨胀 ==========
        hole_dilated, added_pixels = dilate_hole_right_adaptive(
            hole, MAX_DILATE, MAX_DISPARITY, device
        )
        added_pixels_list.append(added_pixels)
        right_with_hole = right_warped.clone()
        right_with_hole[hole_dilated] = 0.0

        # ========== 三阶段填充 ==========
        # 阶段 1：严格右侧填充
        result, hole_after_s1 = inpaint_with_kernel(
            right_with_hole, hole_dilated, kernel_right, k3_right, STAGE1_ITERS
        )
        remaining_after_s1.append(hole_after_s1.sum().item())

        # 阶段 2：垂直方向填充
        result, hole_after_s2 = inpaint_with_kernel(
            result, hole_after_s1, kernel_vert, k3_vert, STAGE2_ITERS
        )
        remaining_after_s2.append(hole_after_s2.sum().item())

        # 阶段 3：全向小核清理
        result, hole_final = inpaint_with_kernel(
            result, hole_after_s2, kernel_full, k3_full, STAGE3_ITERS
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

    print(f"\n[v46 自适应膨胀] ✅ 完成！")
    print(f"  共处理 {len(frame_times)} 帧")
    print(f"  平均耗时: {np.mean(frame_times)*1000:.1f} ms/帧")
    print(f"\n  膨胀统计:")
    print(f"    平均每帧新增像素: {np.mean(added_pixels_list):.0f}")
    print(f"\n  三阶段剩余空洞统计:")
    print(f"    阶段 1 (右向): 平均 {np.mean(remaining_after_s1):.0f} 像素")
    print(f"    阶段 2 (垂直): 平均 {np.mean(remaining_after_s2):.0f} 像素")
    print(f"    阶段 3 (全向): 平均 {np.mean(remaining_final):.0f} 像素")
    print(f"\n  输出视频: {out_path}")
    print(f"  SBS 对比: {sbs_out_path}")


if __name__ == "__main__":
    main()
