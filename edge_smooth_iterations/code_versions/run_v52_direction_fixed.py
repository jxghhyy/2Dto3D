"""
v52 方向修正版：从左向右填充

正确的空洞结构：
- 左视图 warp → 右视图
- 前景物体左移，其左边缘的左侧产生空洞
- 空洞的左侧是背景，右侧是前景物体
- 所以应该：从左向右填充（背景→前景），防渗前景色到背景
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

KERNEL_SIZE = 15


def create_left_to_right_kernel(kernel_size, device):
    """
    从左向右填充的卷积核
    右侧全部为 0，左侧全部为 1
    这样中心像素只能看到自己左侧的内容（背景）
    """
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=device)
    half = kernel_size // 2
    # 水平方向：从 half 到 end 都为 1（即中心的左侧）
    # 这样卷积时，中心像素只能从左侧获取信息
    kernel[0, 0, :, half:] = 1.0
    return kernel


@torch.no_grad()
def dilate_right_gpu(right_warped, hole, max_dilate=8, color_threshold=0.15, device='cuda'):
    """只向右膨胀（v51 修复版）- 这个方向是对的，检测颜色相似的前景毛刺"""
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
            ref_color = right_warped_np[y, start_x - 1]  # 空洞左边界 = 前景右边界
            for shift in range(1, max_dilate + 1):
                check_x = end_x + shift  # 检查空洞右侧（背景侧）的白色毛刺
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
def inpaint_left_to_right(img, hole, kernel, max_iter=8):
    """
    纯从左向右填充（背景→前景）
    防止前景颜色渗到背景
    """
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel.shape[-1] // 2
    k3 = kernel.repeat(3, 1, 1, 1)

    fill_counts = []
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
        fill_count = can_fill.sum().item()
        fill_counts.append(fill_count)

        if fill_count == 0:
            break

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        if not hole_cur.any():
            break

    return result, hole_cur, fill_counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v52 从左向右填充] 设备: {device}")
    print(f"[v52 从左向右填充] 卷积核尺寸: {KERNEL_SIZE}")

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

    OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v52_left_to_right")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kernel = create_left_to_right_kernel(KERNEL_SIZE, device)

    print("\n测试从左向右填充...")
    t0 = time.time()
    result, hole_final, fill_counts = inpaint_left_to_right(
        right_warped, hole_dilated, kernel, 8
    )
    t1 = time.time()
    remaining = hole_final.sum().item()
    print(f"  剩余空洞: {remaining}, 耗时: {(t1-t0)*1000:.1f}ms")
    print(f"  每轮填充: {fill_counts}")

    result_np = (result.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "result.png"), cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))

    left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
    sbs = np.concatenate([left_uint8, result_np], axis=1)
    cv2.imwrite(str(OUT_DIR / "sbs.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    for crop_name, (y1, y2, x1, x2) in [
        ("face", (200, 500, 600, 900)),
        ("shoulder", (550, 750, 750, 1050)),
    ]:
        crop = result_np[y1:y2, x1:x2]
        cv2.imwrite(str(OUT_DIR / f"crop_{crop_name}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 50)
    print("【颜色对比】")
    v52 = cv2.imread(str(OUT_DIR / "crop_shoulder.png"))
    bg_v52 = v52[50:150, 150:250]
    print(f"v52 (从左向右 k15): R={bg_v52[:,:,2].mean():.1f}, G={bg_v52[:,:,1].mean():.1f}, B={bg_v52[:,:,0].mean():.1f}")

    v52_wrong = cv2.imread("frames/v52_kernel_size_test/crop_shoulder_k15.png")
    bg_wrong = v52_wrong[50:150, 150:250]
    print(f"v52 (从右向左 k15): R={bg_wrong[:,:,2].mean():.1f}, G={bg_wrong[:,:,1].mean():.1f}, B={bg_wrong[:,:,0].mean():.1f}")

    print(f"  理想值: 背景 ~90，前景 ~240")
    print(f"  越接近 90 = 渗色越少！")
    print("=" * 50)
    print(f"\n输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
