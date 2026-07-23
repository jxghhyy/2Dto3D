"""
专门分析肩膀区域的空洞
用户说的是下面的肩膀，不是脸！
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
OUT_DIR = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/shoulder_analysis")
VIDEO_PATH = "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton_2min.mp4"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    right_np = (right_warped * 255).byte().cpu().numpy()
    hole_np = hole.cpu().numpy()

    # ========== 肩膀区域：用户说的是下面！ ==========
    y1, y2 = 500, 800  # 向下移，看肩膀
    x1, x2 = 750, 1050

    # 保存原图裁剪
    crop_original = right_np[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / "01_shoulder_original.png"), cv2.cvtColor(crop_original, cv2.COLOR_RGB2BGR))
    print(f"保存肩膀裁剪: {y1}:{y2}, {x1}:{x2}")

    # 保存带空洞标记的裁剪
    crop_with_hole = crop_original.copy()
    crop_hole = hole_np[y1:y2, x1:x2]
    crop_with_hole[crop_hole] = 0  # 空洞涂黑
    cv2.imwrite(str(OUT_DIR / "02_shoulder_with_hole.png"), cv2.cvtColor(crop_with_hole, cv2.COLOR_RGB2BGR))

    # ========== 逐行采样分析空洞右边界 ==========
    print(f"\n【空洞右边界采样分析（肩膀区域）】")
    print(f"  空洞左边界像素 vs 空洞右边界的像素")

    sample_count = 0
    for y in range(y1, y2, 3):  # 每 3 行采样
        row = hole_np[y]
        indices = np.where(row)[0]
        if len(indices) == 0:
            continue

        # 找到所有连续空洞区域
        start = None
        regions = []
        for i in range(len(indices)):
            if start is None:
                start = indices[i]
                prev = indices[i]
            elif indices[i] > prev + 1:
                regions.append((start, prev))
                start = indices[i]
            prev = indices[i]
        if start is not None:
            regions.append((start, prev))

        for start_x, end_x in regions:
            # 只看在裁剪范围内的空洞
            if x1 <= start_x <= x2 and x1 <= end_x <= x2:
                if start_x > 0:
                    ref_pixel = right_np[y, start_x - 1]  # 空洞左边的前景颜色

                    # 向右检测最多 10 个像素
                    for shift in range(1, 11):
                        check_x = end_x + shift
                        if check_x >= w_orig:
                            break

                        pixel = right_np[y, check_x]
                        diff = np.abs(ref_pixel.astype(float) - pixel.astype(float)).mean()

                        if sample_count < 20:  # 打印前 20 个
                            print(f"  y={y}, shift={shift}: 参考({start_x-1})={ref_pixel}, "
                                  f"检测({check_x})={pixel}, Δ={diff:.1f}")
                        sample_count += 1

    print(f"\n  共采样 {sample_count} 个点")

    # ========== v47 膨胀效果用于对比 ==========
    hole_float = hole.float()
    cumsum_from_right = torch.cumsum(hole_float.flip(dims=[1]), dim=1).flip(dims=[1])
    hole_widths = cumsum_from_right * hole_float
    dilate_per_pixel = (hole_widths / 24.0 * 5).round().long()
    dilate_per_pixel = torch.clamp(dilate_per_pixel, 1, 5)
    hole_v47 = hole.clone()
    for shift in range(1, 5 + 1):
        shifted = torch.roll(hole, shifts=shift, dims=1)
        shifted[:, :shift] = False
        should_dilate = (dilate_per_pixel >= shift) & shifted
        hole_v47 = hole_v47 | should_dilate

    hole_v47_np = hole_v47.cpu().numpy()
    crop_v47 = crop_original.copy()
    crop_v47[hole_v47_np[y1:y2, x1:x2]] = 0
    cv2.imwrite(str(OUT_DIR / "03_shoulder_v47_dilate.png"), cv2.cvtColor(crop_v47, cv2.COLOR_RGB2BGR))

    print(f"\n✅ 分析图已保存到: {OUT_DIR}")
    print(f"  请打开 01_shoulder_original.png 查看肩膀区域")


if __name__ == "__main__":
    main()
