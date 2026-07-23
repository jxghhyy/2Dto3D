"""
迭代 v44: 修复多区域膨胀 Bug + 严格右侧填充

核心修复:
  Bug: 原来的膨胀只膨胀了一行中最右边的空洞区域，中间的空洞完全没膨胀！
  Fix: 对一行中的每个连续空洞区域都分别向右膨胀

版本目录: v44/dilate_right{N}_kernel{K}/
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
BASE_OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v44")


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
    """
    ✅ 修复版：对每行中的每个连续空洞区域，分别向右膨胀
    不再只膨胀最右边的那个空洞！
    """
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()

    for y in range(h):
        row = hole_np[y]
        if not row.any():
            continue

        # 找这一行中所有的连续空洞区域的起止点
        indices = np.where(row)[0]
        regions = []  # [(start_x, end_x), ...]

        if len(indices) > 0:
            start_x = indices[0]
            prev_x = indices[0]
            for i in range(1, len(indices)):
                x = indices[i]
                if x > prev_x + 1:
                    # 新区域开始
                    regions.append((start_x, prev_x))
                    start_x = x
                prev_x = x
            # 最后一个区域
            regions.append((start_x, prev_x))

        # 对每个区域，分别向右膨胀
        for (start_x, end_x) in regions:
            new_end_x = min(w - 1, end_x + dilate_pixels)
            hole_dilated[y, end_x:new_end_x + 1] = True

    return torch.from_numpy(hole_dilated).to(device)


@torch.no_grad()
def inpaint_strict_right(img, hole, kernel_size=11, max_iter=40):
    result = img.clone()
    hole_cur = hole.clone()
    pad = kernel_size // 2
    device = hole.device

    kernel = create_strict_right_kernel(kernel_size, device)
    k3 = kernel.repeat(3, 1, 1, 1)

    steps = []

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

        steps.append({
            "iter": it + 1,
            "image": result.clone(),
            "filled_count": filled_count,
            "remaining_hole": hole_cur.sum().item()
        })

        result[can_fill] = avg_hwc[can_fill]
        hole_cur[can_fill] = False

        if not hole_cur.any():
            break

    steps.append({
        "iter": "final",
        "image": result.clone(),
        "filled_count": 0,
        "remaining_hole": hole_cur.sum().item()
    })

    return result, steps


def main_debug_frame():
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v44] 使用设备: {device}")
    print(f"[v44] 输出根目录: {BASE_OUT_DIR}")

    # 加载模型
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(CHECKPOINTS_DIR / "depth_anything_v2_vits.pth"), map_location="cpu"))
    model = model.to(device).eval()

    # 加载第70秒帧
    video_path = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(70 * fps)
    print(f"[v44] 处理第 70 秒 → 第 {target_frame} 帧")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame_bgr = cap.read()
    cap.release()

    h_orig, w_orig = frame_bgr.shape[:2]

    # 深度推理
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

    dibr_h, dibr_w = h_orig, w_orig
    near_score = F.interpolate(
        depth_norm[None, None, :, :],
        size=(dibr_h, dibr_w),
        mode="bilinear", align_corners=False
    )[0, 0]
    max_disparity = 24.0 * dibr_w / w_orig
    disparity = near_score * max_disparity

    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb_tensor = torch.from_numpy(left_rgb_original).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb_tensor, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    print(f"[v44] 原始空洞: {hole.sum().item()} 像素")

    # ========== 先验证修复效果 ==========
    verify_dir = BASE_OUT_DIR / "_verify_dilate_fix"
    verify_dir.mkdir(parents=True, exist_ok=True)

    orig_vis = (right_warped * 255).byte().cpu().numpy().copy()
    orig_vis[hole.cpu().numpy()] = 0
    cv2.imwrite(str(verify_dir / "00_original_hole.png"), cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR))

    for dilate_pixels in [3, 5, 8, 11, 15]:
        hole_dilated = dilate_hole_right_fixed(hole, dilate_pixels, device)
        dilated_np = hole_dilated.cpu().numpy()
        new_pixels = dilated_np & ~hole.cpu().numpy()
        print(f"  [验证] 膨胀 {dilate_pixels} 像素: 新增 {new_pixels.sum()} 像素 (v43只有400左右)")

        # 高亮差异
        diff_vis = (right_warped * 255).byte().cpu().numpy().copy()
        diff_vis[new_pixels] = [255, 0, 0]  # 红色高亮新增像素
        diff_vis[hole.cpu().numpy()] = 0     # 原始空洞保持黑色
        cv2.imwrite(str(verify_dir / f"01_dilate_{dilate_pixels}_new_red.png"),
                   cv2.cvtColor(diff_vis, cv2.COLOR_RGB2BGR))

        # 裁剪中间区域放大
        x1, x2 = 900, 1000
        y1, y2 = 300, 800
        crop = diff_vis[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        crop_bgr_x3 = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(verify_dir / f"02_crop_{dilate_pixels}_x3.png"), crop_bgr_x3)

    # ========== 开始正式测试 ==========
    for dilate_pixels in [0, 3, 5, 8, 11]:
        if dilate_pixels == 0:
            hole_dilated = hole
        else:
            hole_dilated = dilate_hole_right_fixed(hole, dilate_pixels, device)
        dilated_count = hole_dilated.sum().item()
        print(f"\n[v44] 向右膨胀 {dilate_pixels} 像素 → {dilated_count:,} 像素 "
              f"(+{dilated_count - hole.sum().item():,})")

        for kernel_size in [11, 15]:
            subdir = BASE_OUT_DIR / f"dilate_right{dilate_pixels}_kernel{kernel_size}"
            subdir.mkdir(parents=True, exist_ok=True)

            # 保存膨胀后的空洞可视化
            right_with_hole = right_warped.clone()
            right_with_hole[hole_dilated] = 0.0
            hole_vis = (right_with_hole * 255).byte().cpu().numpy()
            cv2.imwrite(str(subdir / "00_hole_after_dilate.png"),
                       cv2.cvtColor(hole_vis, cv2.COLOR_RGB2BGR))

            # 严格右侧填充
            print(f"       核 {kernel_size}×{kernel_size} 填充...")
            t0 = time.time()
            result, steps = inpaint_strict_right(
                right_with_hole, hole_dilated,
                kernel_size=kernel_size, max_iter=50
            )
            t_v44 = time.time() - t0
            print(f"         耗时: {t_v44*1000:.1f} ms, {len(steps)-1} 次迭代")
            print(f"         剩余空洞: {steps[-1]['remaining_hole']:,} 像素")

            for i, step in enumerate(steps):
                img_np = (step["image"].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(subdir / f"iter_{i:02d}_{step['iter']}.png"),
                           cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            left_uint8 = (left_rgb_tensor * 255).byte().cpu().numpy()
            right_uint8 = (result * 255).byte().cpu().numpy()
            sbs = np.concatenate([left_uint8, right_uint8], axis=1)
            cv2.imwrite(str(subdir / "final_sbs.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

    print(f"\n[v44] ✅ 所有测试完成！")


if __name__ == "__main__":
    main_debug_frame()
