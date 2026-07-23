"""
验证向右膨胀是否真的生效
生成对比图：红色高亮 = 新膨胀出来的像素
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


def dilate_hole_right(hole, dilate_pixels, device):
    """逐行向右膨胀"""
    h, w = hole.shape
    hole_np = hole.cpu().numpy()
    hole_dilated = hole_np.copy()

    for y in range(h):
        row = hole_np[y]
        if not row.any():
            continue
        right_x = np.where(row)[0][-1]
        new_right_x = min(w - 1, right_x + dilate_pixels)
        hole_dilated[y, right_x:new_right_x + 1] = True

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

    out_dir = Path("/mnt/A/jiangxg/work/2Dto3D/edge_smooth_iterations/frames/verify_dilate")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 原始空洞
    hole_np = hole.cpu().numpy()
    orig_vis = (right_warped * 255).byte().cpu().numpy().copy()
    orig_vis[hole_np] = 0
    cv2.imwrite(str(out_dir / "01_original_hole.png"), cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR))

    print(f"原始空洞: {hole_np.sum()} 像素")

    # 测试不同膨胀量
    for dilate_pixels in [3, 5, 8, 11, 15]:
        hole_dilated = dilate_hole_right(hole, dilate_pixels, device)
        dilated_np = hole_dilated.cpu().numpy()

        # 高亮差异：新膨胀出来的像素用红色标记
        diff_vis = (right_warped * 255).byte().cpu().numpy().copy()

        new_pixels = dilated_np & ~hole_np  # 膨胀新增的像素
        print(f"膨胀 {dilate_pixels} 像素: 新增 {new_pixels.sum()} 像素")

        diff_vis[new_pixels] = [255, 0, 0]  # 红色高亮新增像素
        diff_vis[hole_np] = 0               # 原始空洞保持黑色

        cv2.imwrite(str(out_dir / f"02_dilate_{dilate_pixels}_new_pixels_red.png"),
                   cv2.cvtColor(diff_vis, cv2.COLOR_RGB2BGR))

        # 放大裁剪图：聚焦空洞左边界附近（x=700~900）
        crop_x1, crop_x2 = 700, 900
        crop_y1, crop_y2 = 200, 400
        crop_diff = diff_vis[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_diff_bgr = cv2.cvtColor(crop_diff, cv2.COLOR_RGB2BGR)
        crop_diff_bgr = cv2.resize(crop_diff_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"03_crop_dilate_{dilate_pixels}_x2.png"), crop_diff_bgr)

    print(f"\n✅ 验证图已保存到: {out_dir}")
    print(f"  红色 = 膨胀新增的像素")
    print(f"  黑色 = 原始空洞")


if __name__ == "__main__":
    main()
