"""
修复 side-by-side 对比视频生成
将四阶段视频正确合成为 2x2 网格（而不是 4 个并排）
"""
import cv2
import numpy as np
from pathlib import Path

# 输入视频路径
base_dir = Path("edge_smooth_iterations/frames/v29_fixed_full")
videos = [
    ("v29_01_warp_only.mp4", "Warp Only (Red Hole)"),
    ("v29_02_with_band.mp4", "With Occlusion Band"),
    ("v29_03_inpaint_only.mp4", "Inpaint Only"),
    ("v29_04_final_complete.mp4", "Final Complete"),
]

# 打开所有视频
caps = []
for fname, _ in videos:
    cap = cv2.VideoCapture(str(base_dir / fname))
    caps.append(cap)
    print(f"✅ 打开: {fname} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} 帧)")

# 获取视频信息
width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = caps[0].get(cv2.CAP_PROP_FPS)
total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 视频信息: {width}x{height}, {fps} fps, {total_frames} 帧")

# 2x2 网格输出尺寸
out_width = width * 2
out_height = height * 2
print(f"📐 输出尺寸: {out_width}x{out_height} (2x2 网格)")

# 创建视频写入器 (使用 libx264 CPU 编码，避免 NVENC 限制)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = str(base_dir / "v29_05_side_by_side_2x2.mp4")
out = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))

# 字体设置
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
font_thickness = 3
text_color = (255, 255, 255)  # 白色
bg_color = (0, 0, 0)  # 黑色背景

print(f"\n🎬 开始生成 2x2 对比视频...")

for frame_idx in range(total_frames):
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(frame)

    # 构建 2x2 网格
    row1 = np.hstack([frames[0], frames[1]])
    row2 = np.hstack([frames[2], frames[3]])
    grid = np.vstack([row1, row2])

    # 添加标签
    positions = [
        (10, 40),          # 左上
        (width + 10, 40),  # 右上
        (10, height + 40), # 左下
        (width + 10, height + 40),  # 右下
    ]

    for (_, label), pos in zip(videos, positions):
        # 文字背景
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(grid, (pos[0] - 5, pos[1] - text_h - 5),
                     (pos[0] + text_w + 5, pos[1] + 5), bg_color, -1)
        cv2.putText(grid, label, pos, font, font_scale, text_color, font_thickness)

    out.write(grid)

    if (frame_idx + 1) % 50 == 0:
        print(f"  处理: {frame_idx + 1}/{total_frames} 帧")

# 清理
for cap in caps:
    cap.release()
out.release()

print(f"\n✅ 完成! 输出视频: {out_path}")
print(f"   文件大小: {Path(out_path).stat().st_size / 1024 / 1024:.1f} MB")
