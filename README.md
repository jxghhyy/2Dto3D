# 2D 转 3D 视频处理

DepthAnythingV2 + GPU DIBR + FastInpaint 的 2D 转 3D 立体视频 pipeline

## 📁 版本说明

| 文件 | 说明 |
|------|------|
| `mono23d.py` | xz 师兄的 fast-inpaint 方法 |
| `mono2stereo_lowres.py` | zc 师兄的降分辨率处理后再上采样的方法 |
| `mono2stereo_lower_fastinpaint.py` | 在 xz 师兄框架上加入降分辨率上采样优化 |
| `mono2stereo_lower_fastinpaint_time.py` | 增加 `--profile-time` 参数，计时 + 界面优化 |
| `mono2stereo_lower_fastinpaint_time_gpu.py` | ✅ **最新 GPU 全优化版本** |

> ⚠️ zc 师兄的方法未统计 OpenCV 写视频时间，因此对比时使用 `mono23d.py` 和 `mono2stereo_lower_fastinpaint.py`

## 🚀 性能对比

### 本地测试
- `mono23d.py` (纯 fast-inpaint): **16.6 FPS**
- `mono2stereo_lower_fastinpaint.py` (降采样优化): **26.1 FPS** ✨
- **提升**: +57%

### 民大服务器 3090 (第7张卡)
- `mono2stereo_lower_fastinpaint.py`: **~10 FPS**
  - 推测瓶颈: ffmpeg 编码格式、CPU 调用开销
- `mono2stereo_lower_fastinpaint_time_gpu.py` (全 GPU 优化): **16 FPS** ✨
- **提升**: +60%

## ⚡ GPU 优化内容

`mono2stereo_lower_fastinpaint_time_gpu.py` 包含以下优化:

1. **颜色转换移到 GPU** - BGR→RGB 不再占用 CPU
2. **多线程预读队列** - 后台线程预读帧，CPU-GPU 并行工作
3. **预处理全 GPU 化** - Resize + Normalize 全部在 GPU 完成
   - 比 CPU 版本快 **6-10 倍**

## 📊 性能分析

运行时添加 `--profile-time` 参数即可查看各阶段耗时统计:

```bash
python mono2stereo_lower_fastinpaint_time_gpu.py \
  --video-path your_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --profile-time
```

输出包含:
- 顶层阶段耗时占比
- 各模块内部细分明细
- 平均 FPS
- 预读队列效率判断

## 🔧 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video-path` | 输入视频路径、目录或 txt 列表 | ✅ 必填 |
| `--input-size` | DepthAnythingV2 长边像素（保持宽高比，必须 14 倍数） | 518 |
| `--encoder` | 模型规模: vits/vitb/vitl/vitg | vits |
| `--max-disparity` | 低分辨率下最大水平视差像素 | 16.0 |
| `--fp16` | 启用 FP16 推理（速度更快） | ❌ |
| `--profile-time` | 输出各阶段耗时统计 | ❌ |
| `--layout` | 立体布局: sbs(并排)/ou(上下)/overlay | sbs |

## 💡 提示

- 如果 `read_frame_queue` 耗时接近 0 → 预读队列工作正常，GPU 没有在等 CPU
- 推荐开启 `--fp16`，3090 上速度再提升 ~30%
- `--input-size` 现在保持宽高比，不再强制正方形！（长边固定，另一边按比例缩放）
