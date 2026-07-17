"""
调试版本：看看原始 B 版本的反遮挡带长什么样
不做任何后过滤
"""
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, '.')
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
import mono2stereo_b as b

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--encoder", type=str, default="vits")
    return parser.parse_args()


def project_disocclusion_bands_original(disparity, min_drop=3.0, right_cleanup=16):
    """完全和 B 版本一样，不做任何修改"""
    h, w = disparity.shape
    if w < 2:
        return torch.zeros_like(disparity, dtype=torch.bool)

    device = disparity.device
    x_left = torch.arange(w - 1, device=device).view(1, w - 1)
    d_left = disparity[:, :-1]
    d_right = disparity[:, 1:]
    is_drop = (d_left - d_right) >= min_drop

    foreground_target = x_left.to(disparity.dtype) - d_left
    background_target = (x_left + 1).to(disparity.dtype) - d_right
    start = torch.floor(foreground_target).long() + 1
    end = torch.floor(background_target).long() + right_cleanup
    start = start.clamp(0, w - 1)
    end = end.clamp(0, w - 1)
    valid = is_drop & (end >= start)

    difference = torch.zeros((h, w + 1), device=device, dtype=torch.int32)
    rows = torch.arange(h, device=device).view(h, 1).expand(h, w - 1)
    flat = difference.reshape(-1)
    start_index = (rows * (w + 1) + start)[valid]
    stop_index = (rows * (w + 1) + end + 1)[valid]
    flat.scatter_add_(0, start_index, torch.ones_like(start_index, dtype=flat.dtype))
    flat.scatter_add_(0, stop_index, -torch.ones_like(stop_index, dtype=flat.dtype))

    band = torch.cumsum(difference[:, :w], dim=1) > 0
    return band


def process_frame(frame_bgr, model, device, args):
    h_orig, w_orig = frame_bgr.shape[:2]

    # 深度推理
    img = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)
    img_resized = F.interpolate(img, size=(294, 518), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    model_input = (img_resized - mean_t) / std_t

    with torch.no_grad():
        depth_raw = model(model_input)[0].float()

    # 归一化
    flat = depth_raw.reshape(-1)
    idx = torch.randint(0, flat.numel(), (16384,), device=flat.device)
    sample = flat[idx]
    q_vals = torch.quantile(sample, torch.tensor([0.01, 0.99], device=flat.device))
    low, high = q_vals[0], q_vals[1]
    depth_norm = ((depth_raw - low) / (high - low)).clamp(0.0, 1.0)

    # 上采样
    near_score = F.interpolate(
        depth_norm[None, None, :, :],
        size=(h_orig, w_orig),
        mode="bilinear", align_corners=False
    )[0, 0]

    disparity = near_score * 24.0

    # B版本：锐化
    disparity_sharp, unreliable = b.sharpen_disparity_edges(
        disparity, kernel_size=15, threshold=3.0, iterations=1, reject_margin=0.10
    )

    left_rgb = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(device).float() / 255.0
    right_warped, hole = b.forward_warp_excluding_source(
        left_rgb, disparity_sharp, near_score, unreliable, {}, False, 0.01
    )

    # 原始反遮挡带
    band_original = project_disocclusion_bands_original(
        disparity_sharp, min_drop=3.0, right_cleanup=16
    )

    return {
        'band_original': band_original.cpu().numpy(),
        'hole': hole.cpu().numpy(),
        'warped': right_warped.cpu().numpy(),
        'disparity': disparity_sharp.cpu().numpy(),
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 调试原始反遮挡带")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = DepthAnythingV2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384])
    ckpt = f"./checkpoints/depth_anything_v2_{args.encoder}.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    # 处理几帧
    test_frames = [60, 90, 210, 240]

    for frame_idx in test_frames:
        cap = cv2.VideoCapture(args.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        cap.release()

        if not ok:
            continue

        print(f"\n处理帧 {frame_idx}...")
        result = process_frame(frame_bgr, model, device, args)

        h, w = frame_bgr.shape[:2]

        # 可视化：在 warp 图上把反遮挡带标成绿色
        band_viz = (result['warped'] * 255).astype(np.uint8)
        band_viz[result['band_original']] = [0, 255, 0]
        cv2.imwrite(str(outdir / f'debug_frame_{frame_idx:03d}_band_green.png'),
                   cv2.cvtColor(band_viz, cv2.COLOR_RGB2BGR))

        # 画空洞边界
        hole_viz = (result['warped'] * 255).astype(np.uint8)
        hole_viz[result['hole']] = [0, 0, 255]
        cv2.imwrite(str(outdir / f'debug_frame_{frame_idx:03d}_hole_red.png'),
                   cv2.cvtColor(hole_viz, cv2.COLOR_RGB2BGR))

        # 视差图可视化
        disp_norm = cv2.normalize(result['disparity'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(outdir / f'debug_frame_{frame_idx:03d}_disparity.png'), disp_color)

        print(f"  反遮挡带像素: {result['band_original'].sum():,}")
        print(f"  空洞像素: {result['hole'].sum():,}")

    print(f"\n✅ 调试完成！结果在: {outdir}")


if __name__ == "__main__":
    main()
