import torch
import cv2
import time
import subprocess
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
from submodules.depth.dav2.depth_anything_v2.util.transform import MyResize, NormalizeImage, PrepareForNet


def fast_dibr_pytorch(left_img_tensor, depth_tensor, max_disparity=30.0):
    """
    纯 PyTorch 实现的极速前向翘曲 (Forward Warping) - 修正版
    """
    B, C, H, W = left_img_tensor.shape
    device = left_img_tensor.device

    # 1. 深度归一化 (0~1)
    d_min = depth_tensor.min()
    d_max = depth_tensor.max()
    norm_depth = (depth_tensor - d_min) / (d_max - d_min + 1e-5)

    # 2. 计算视差偏移量
    disparity = norm_depth * max_disparity

    # 3. 生成像素网格坐标
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    y = y.expand(B, -1, -1)
    x = x.expand(B, -1, -1)

    # 4. 计算右眼图中的新 X 坐标
    new_x = x - disparity.squeeze(1) if disparity.dim() == 4 else x - disparity
    new_x = torch.round(new_x).long()

    # 过滤掉飞出画面边界的像素
    valid_mask = (new_x >= 0) & (new_x < W)

    # 5. 准备 Z-Buffer (深度测试)
    right_img = torch.zeros_like(left_img_tensor)

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)[valid_mask]
    y_idx = y[valid_mask]
    new_x_idx = new_x[valid_mask]

    valid_depth = norm_depth.squeeze(1)[valid_mask]
    valid_color = left_img_tensor.permute(0, 2, 3, 1)[valid_mask]

    # 按深度升序排序 (从远到近)，后写入覆盖前写入，实现"近遮远"
    sorted_indices = torch.argsort(valid_depth, descending=False)

    b_idx_sorted = b_idx[sorted_indices]
    y_idx_sorted = y_idx[sorted_indices]
    new_x_idx_sorted = new_x_idx[sorted_indices]
    color_sorted = valid_color[sorted_indices]

    right_img[b_idx_sorted, :, y_idx_sorted, new_x_idx_sorted] = color_sorted

    # 6. 生成 Mask (标记哪些像素被填充了，哪些是黑洞)
    mask = torch.zeros((B, 1, H, W), device=device, dtype=torch.uint8)
    mask[b_idx_sorted, 0, y_idx_sorted, new_x_idx_sorted] = 255

    return right_img, mask


def main(video_path, output_path):
    # ---------------------------------------------------------
    # 1. 基础设置与模型加载 (FP16 加速)
    # ---------------------------------------------------------
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print("加载 DepthAnything 模型中..., device: ", device)

    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load('submodules/depth/dav2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
    depth_model = depth_model.to(device).half().eval()  # 开启 FP16

    # 低分辨率工作尺寸（与 DepthAnythingV2 输入一致，必须是 14 的倍数）
    LOW_RES = 518

    transform = Compose([
        MyResize(
            width=LOW_RES, height=LOW_RES,
            resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14,
            resize_method='minimal'
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # ---------------------------------------------------------
    # 2. CUDA 预热
    # ---------------------------------------------------------
    print(" 执行 CUDA 预热...")
    dummy_input = torch.randn(1, 3, LOW_RES, LOW_RES, dtype=torch.float16, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = depth_model(dummy_input)
    torch.cuda.synchronize()
    print("✅ 预热完成！\n")

    # ---------------------------------------------------------
    # 3. 视频流读取与保存设置
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(video_path)

    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    if fps_orig == 0:
        fps_orig = 30.0
    print("w, h: ", w_orig, " ", h_orig, " fps: ", fps_orig)

    # 3D 效果强度：设定最终画面上视差约为原宽的 3%
    # 因为横向从 w_orig 被压到 LOW_RES，然后又上采回 w_orig，
    # 所以在低分辨率下只需要 LOW_RES * 0.03 就能保证最终视觉等价。
    max_disparity_pixels = LOW_RES * 0.03

    out_width = w_orig * 2
    out_height = h_orig

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_orig, (out_width, out_height))

    if not cap.isOpened():
        print(f"❌ 无法打开视频流: {video_path}")
        return

    # 分阶段 FPS 统计容器
    fps_stats = {'depth': [], 'dibr': [], 'inpaint': [], 'upsample': [], 'total': []}

    # 小工具：GPU 阶段计时前后都要 synchronize，保证时间不被"顺延"到后面的同步点
    def _sync_now():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _sync_now()
        t_total_start = time.time()

        # ---------------------------------------------------------
        # [步骤 A] 预处理 & 低分辨率深度估计  (GPU 阶段)
        # ---------------------------------------------------------
        t_a_start = t_total_start
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform({'image': img_rgb / 255.0})['image']
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device).half()  # [1,3,LOW_RES,LOW_RES]

        with torch.no_grad():
            depth = depth_model(img_tensor)  # 输出在 LOW_RES 分辨率

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # -> [1,1,LOW_RES,LOW_RES]

        # 若模型输出尺寸与 LOW_RES 不一致（例如 ensure_multiple_of 微调导致），
        # 做一次轻量对齐（通常已经一致，开销可忽略）
        if depth.shape[-2] != LOW_RES or depth.shape[-1] != LOW_RES:
            depth = F.interpolate(depth.float(), size=(LOW_RES, LOW_RES),
                                  mode='bilinear', align_corners=False)
        else:
            depth = depth.float()
        _sync_now()
        t_a_end = time.time()

        # ---------------------------------------------------------
        # [步骤 B] 低分辨率原图（BGR）送入 GPU 做 DIBR  (GPU 阶段)
        # ---------------------------------------------------------
        t_b_start = t_a_end
        # 直接用 OpenCV 把原始 BGR 帧 resize 到 LOW_RES（这里 keep_aspect_ratio=False，与 transform 一致）
        frame_low = cv2.resize(frame, (LOW_RES, LOW_RES), interpolation=cv2.INTER_LINEAR)

        frame_low_t = torch.from_numpy(frame_low).to(device).float()  # [H,W,3] BGR
        frame_low_t = frame_low_t.permute(2, 0, 1).unsqueeze(0)       # [1,3,H,W]

        with torch.no_grad():
            right_img_t, mask_t = fast_dibr_pytorch(
                frame_low_t, depth, max_disparity=max_disparity_pixels
            )
        _sync_now()
        t_b_end = time.time()

        # ---------------------------------------------------------
        # [步骤 C] 低分辨率下用 Telea 填补黑洞  (CPU 阶段，含 D2H 拷贝)
        # ---------------------------------------------------------
        t_c_start = t_b_end
        right_img_np = right_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask_t.squeeze().cpu().numpy()

        # 翻转 Mask：255 表示黑洞区域，0 表示已有颜色的安全区域
        hole_mask = (255 - mask_np).astype(np.uint8)
        right_inpainted_low = cv2.inpaint(
            right_img_np, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )
        t_c_end = time.time()

        # ---------------------------------------------------------
        # [步骤 D] 左右眼同时双线性上采样到原始尺寸  (CPU 阶段)
        # （左眼也使用 frame_low 上采样，保证左右两眼视觉一致性）
        # ---------------------------------------------------------
        t_d_start = t_c_end
        left_up = cv2.resize(frame_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        right_up = cv2.resize(right_inpainted_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        t_d_end = time.time()

        t_total_end = t_d_end

        # ---------------------------------------------------------
        # 计算各阶段 FPS
        # ---------------------------------------------------------
        fps_a = 1.0 / max(t_a_end - t_a_start, 1e-6)
        fps_b = 1.0 / max(t_b_end - t_b_start, 1e-6)
        fps_c = 1.0 / max(t_c_end - t_c_start, 1e-6)
        fps_d = 1.0 / max(t_d_end - t_d_start, 1e-6)
        fps_total = 1.0 / max(t_total_end - t_total_start, 1e-6)

        fps_stats['depth'].append(fps_a)
        fps_stats['dibr'].append(fps_b)
        fps_stats['inpaint'].append(fps_c)
        fps_stats['upsample'].append(fps_d)
        fps_stats['total'].append(fps_total)

        print(f"[Depth {fps_a:6.1f} | DIBR {fps_b:6.1f} | Inpaint {fps_c:6.1f} | "
              f"Upsample {fps_d:6.1f} | Total {fps_total:6.1f}] FPS  | holes={np.count_nonzero(hole_mask)}")

        # ---------------------------------------------------------
        # 拼接 SBS + 在画面上叠加各阶段 FPS
        # ---------------------------------------------------------
        combined_frame = np.hstack([left_up, right_up])
        overlay_texts = [
            f"Total:    {fps_total:6.1f} FPS",
            f"Depth:    {fps_a:6.1f} FPS",
            f"DIBR:     {fps_b:6.1f} FPS",
            f"Inpaint:  {fps_c:6.1f} FPS",
            f"Upsample: {fps_d:6.1f} FPS",
        ]
        for i, text in enumerate(overlay_texts):
            cv2.putText(combined_frame, text, (20, 45 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        out.write(combined_frame)

    cap.release()
    out.release()
    subprocess.run(['ffmpeg', '-y', '-i', output_path, '-vcodec', 'libx264',
                    output_path.split('.mp4')[0] + '_web.mp4'])

    if fps_stats['total']:
        print("\n========== 测试结束 ==========")
        n = len(fps_stats['total'])
        print(f" 总帧数: {n}")
        for name in ['depth', 'dibr', 'inpaint', 'upsample', 'total']:
            arr = fps_stats[name]
            avg = sum(arr) / len(arr)
            # 去掉前 3 帧的热身/JIT 抖动，再算一个稳态均值
            steady = arr[3:] if len(arr) > 6 else arr
            steady_avg = sum(steady) / len(steady)
            print(f" {name:<9s} 平均 FPS: {avg:7.1f}  |  稳态(跳过前3帧) FPS: {steady_avg:7.1f}")


if __name__ == '__main__':
    video_path = '../../datasets/shuai/horse.mp4'
    output_path = 'output/output1.mp4'
    main(video_path, output_path)
