"""
批量运行 v30 版本处理多个视频
每个视频独立输出到自己的目录
"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, '.')

# 要处理的视频列表
videos = [
    "/mnt/A/jiangxg/dataset/shuai/Huawei/spider.mp4",
    "/mnt/A/jiangxg/dataset/shuai/Huawei/CL.mp4",
    "/mnt/A/jiangxg/dataset/shuai/Huawei/badminton.mp4",
    "/mnt/A/jiangxg/dataset/shuai/Huawei/table_tennis.mp4",
]

v30_script = "edge_smooth_iterations/code_versions/iter_v30_early_stop_full.py"
output_base = Path("edge_smooth_iterations/frames/v30_batch")

print("=" * 70)
print("🎬 v30 批量处理开始")
print("=" * 70)
print(f"共 {len(videos)} 个视频")
print(f"输出根目录: {output_base}")
print("=" * 70)

for i, video_path in enumerate(videos):
    video_name = Path(video_path).stem
    outdir = output_base / f"v30_{video_name}"

    print(f"\n\n[{i+1}/{len(videos)}] 处理: {video_name}")
    print("-" * 70)
    print(f"输入: {video_path}")
    print(f"输出: {outdir}")
    print("-" * 70)

    cmd = [
        sys.executable, v30_script,
        "--video-path", video_path,
        "--outdir", str(outdir),
    ]

    print(f"命令: {' '.join(cmd)}")
    print()

    # 运行
    result = subprocess.run(cmd, cwd="/mnt/A/jiangxg/work/2Dto3D")

    if result.returncode == 0:
        print(f"✅ {video_name} 处理完成!")
    else:
        print(f"❌ {video_name} 处理失败 (退出码: {result.returncode})")

print("\n" + "=" * 70)
print("🏁 所有视频处理完成!")
print(f"输出目录: {output_base}/")
print("=" * 70)
