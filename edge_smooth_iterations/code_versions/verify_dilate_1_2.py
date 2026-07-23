"""
验证 dilate=1 和 dilate=2 的膨胀效果
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

    out_dir = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/v44_dilate_1_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_hole_np = hole.cpu().numpy()

    print(f"原始空洞: {orig_hole_np.sum()} 像素")

    for dilate_pixels in [1, 2]:
        hole_dilated = dilate_hole_right_fixed(hole, dilate_pixels, device)
        dilated_np = hole_dilated.cpu().numpy()
        new_pixels = dilated_np & ~orig_hole_np
        print(f"dilate={dilate_pixels}: 新增 {new_pixels.sum()} 像素")

        # 1. 纯空洞图（黑色=空洞）
        hole_vis = (right_warped * 255).byte().cpu().numpy().copy()
        hole_vis[dilated_np] = 0
        cv2.imwrite(str(out_dir / f"hole_after_dilate_{dilate_pixels}.png"),
                   cv2.cvtColor(hole_vis, cv2.COLOR_RGB2BGR))

        # 2. 差异图（红色=新增膨胀像素）
        diff_vis = (right_warped * 255).byte().cpu().numpy().copy()
        diff_vis[new_pixels] = [255, 0, 0]
        diff_vis[orig_hole_np] = 0
        cv2.imwrite(str(out_dir / f"dilate_{dilate_pixels}_new_red.png"),
                   cv2.cvtColor(diff_vis, cv2.COLOR_RGB2BGR))

        # 3. 裁剪中间区域放大 x4
        x1, x2 = 900, 1000
        y1, y2 = 300, 800
        crop = diff_vis[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        crop_bgr_x4 = cv2.resize(crop_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"crop_{dilate_pixels}_x4.png"), crop_bgr_x4)

        # 4. 裁剪球拍区域放大 x4
        x1, x2 = 780, 880
        y1, y2 = 200, 400
        crop2 = diff_vis[y1:y2, x1:x2]
        crop2_bgr = cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR)
        crop2_bgr_x4 = cv2.resize(crop2_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"crop_racket_{dilate_pixels}_x4.png"), crop2_bgr_x4)

    print(f"\n✅ 验证图已保存到: {out_dir}")
    print(f"  hole_after_dilate_1.png - dilate=1 后的空洞")
    print(f"  hole_after_dilate_2.png - dilate=2 后的空洞")
    print(f"  dilate_1_new_red.png - dilate=1 新增像素红色高亮")
    print(f"  dilate_2_new_red.png - dilate=2 新增像素红色高亮")
    print(f"  crop_1_x4.png - 中间区域放大 x4")
    print(f"  crop_2_x4.png - 中间区域放大 x4")
    print(f"  crop_racket_1_x4.png - 球拍区域放大 x4")
    print(f"  crop_racket_2_x4.png - 球拍区域放大 x4")


if __name__ == "__main__":
    main()
