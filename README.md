# 2D 转 3D 立体视频生成

**DepthAnythingV2 + GPU DIBR + FastInpaint** —— 单目视频转高质量立体视频

> ✅ **支持两种时序一致性方案**：光流后处理平滑 / Video-Depth-Anything 视频模型

---

## 🎬 效果预览

| 问题 | 解决前 | 解决后 |
|------|--------|--------|
| **深度抖动** | 帧间深度跳变，闪烁严重 | ✅ 时序一致，平滑过渡 |
| **拉伸变形** | 强制正方形，画面拉伸 | ✅ 保持宽高比，无变形 |

---

## 📦 版本对比

| 文件 | 深度模型 | 时序一致性 | 速度 | 推荐度 |
|------|----------|------------|------|--------|
| `mono23d.py` | DepthAnythingV1 单帧 | ❌ 无 | 慢 | ❌ 旧版 |
| `mono2stereo_lowres.py` | 降分辨率单帧 | ❌ 无 | 中 | ❌ 旧版 |
| `mono2stereo_lower_fastinpaint_time_gpu.py` | DepthAnythingV2 单帧 | ✅ 光流对齐后处理 | 快 | ⭐⭐⭐ |
| `mono2stereo_lower_fastinpaint_time_gpu_video.py` | **Video-Depth-Anything** | ✅✅ 模型级原生时序 | 中 | ⭐⭐⭐⭐⭐ **推荐** |

---

## 🚀 快速开始

### 方式1：视频模型模式（推荐，原生时序一致性）

Video-Depth-Anything 官方流式推理，模型训练时就注入时序监督，**无需后处理平滑！**

```bash
python mono2stereo_lower_fastinpaint_time_gpu_video.py \
  --video-path your_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --video-model  # ← 核心！开启视频模型
```

### 方式2：单帧 + 光流平滑（无额外权重）

用普通 DepthAnythingV2 单帧模型 + 我们的光流对齐平滑：

```bash
python mono2stereo_lower_fastinpaint_time_gpu_video.py \
  --video-path your_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --depth-smooth 0.3  # ← 开启光流平滑，建议 0.2~0.5
```

### 性能分析模式

添加 `--profile-time` 查看各阶段耗时：

```bash
python mono2stereo_lower_fastinpaint_time_gpu_video.py \
  --video-path your_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --video-model \
  --profile-time  # ← 打印性能分析
```

---

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video-path` | 输入视频路径、目录或 txt 列表 | ✅ 必填 |
| `--output` | 输出视频路径（单文件模式） | `./output.mp4` |
| `--input-size` | 深度模型长边像素（必须 14 倍数） | 518 |
| `--encoder` | 模型规模: `vits` / `vitb` / `vitl` | vits |
| `--max-disparity` | 低分辨率下最大视差像素（控制 3D 强度） | 16.0 |
| `--fp16` | 启用 FP16 推理（速度提升 30%+） | ❌ |
| `--video-model` | 使用 Video-Depth-Anything 视频模型（内置时序） | ❌ |
| `--metric` | 视频模型使用 Metric 深度（需要对应权重） | ❌ |
| `--depth-smooth` | 光流时序平滑强度（单帧模式用，0=关闭） | 0.0 |
| `--layout` | 立体布局: `sbs` 并排 / `ou` 上下 / `overlay` 重合 | sbs |
| `--profile-time` | 输出各阶段耗时统计 | ❌ |

---

## 💡 3D 强度调整

主要调 `--max-disparity` 参数：

| 值 | 效果 | 适用场景 |
|----|------|----------|
| 8~12 | 浅 3D，出屏效果弱 | 纪录片、风景 |
| **16** | 平衡，默认推荐 | 一般视频 |
| 20~24 | 强 3D，出屏明显 | 人物特写、动画 |

> 提示：不要超过 32，否则会有明显伪影

---

## 📊 性能对比（3090）

以 1920×1080 视频为例：

| 模式 | FPS | 说明 |
|------|-----|------|
| 单帧模型（无平滑） | ~20 | 速度最快 |
| 单帧模型 + 光流平滑 | ~17 | + 几毫秒光流计算 |
| **VDA 视频模型** | ~12 | 模型自带时序注意力，效果最好 |

性能瓶颈：
- 🔴 深度推理（最耗时）
- 🟡 DIBR 像素重投影
- 🟢 ffmpeg 编码（GPU 加速，很快）

---

## 📦 权重下载

### DepthAnythingV2 单帧模型
```
checkpoints/depth_anything_v2_vits.pth
checkpoints/depth_anything_v2_vitb.pth
checkpoints/depth_anything_v2_vitl.pth
```
下载地址：https://github.com/LiheYoung/Depth-Anything

### Video-Depth-Anything 视频模型
```
checkpoints/video_depth_anything_vits.pth
checkpoints/video_depth_anything_vitb.pth
checkpoints/video_depth_anything_vitl.pth
checkpoints/metric_video_depth_anything_vits.pth  # Metric 版本
```
下载地址：https://github.com/DepthAnything/Video-Depth-Anything

---

## 🔧 环境依赖

```bash
pip install torch opencv-python numpy torchvision einops
```

ffmpeg（需要支持 h264_nvenc）：
```bash
# Ubuntu
sudo apt install ffmpeg

# 或者编译支持 NVENC 的版本
```

---

## 🎯 技术亮点

### 1. ✅ 保持宽高比
- 不再强制正方形，自动根据原视频比例计算目标尺寸
- 确保两个维度都是 14 的倍数（模型要求）
- 无拉伸、无变形

### 2. ✅ 时序一致性（两种方案）

**A. 光流对齐后处理**
- 计算帧间光流
- 把上一帧深度 warp 到当前视角
- 再做加权平滑
- 优点：无额外权重，速度快

**B. Video-Depth-Anything 视频模型**
- 官方流式推理 API
- 模型内部维护时序状态
- 训练时就注入时序监督
- 优点：原生级一致性，运动场景也不糊

### 3. ✅ 全 GPU 加速
- 预处理（Resize + Normalize）全部在 GPU
- BGR→RGB 颜色转换 GPU 加速
- DIBR 像素重投影 GPU 并行
- FastInpaint 补洞 GPU 卷积
- 多线程预读队列，CPU-GPU 流水线并行

---

## 🐛 常见问题

### Q: 报错 `ModuleNotFoundError: No module named 'einops'`
```bash
pip install einops
```

### Q: 视频模型导入失败
确认 `submodules/Video_Depth_Anything` 目录存在，脚本开头已经自动加了路径。

### Q: 3D 效果太强/太弱
调 `--max-disparity`：
- 减小 → 3D 更浅
- 增大 → 3D 更强

### Q: 左右眼反了
简单，把 `compose_stereo` 里的顺序换一下：
```python
# 原来是
return np.concatenate([left_u8, right_u8], axis=1)
# 换成
return np.concatenate([right_u8, left_u8], axis=1)
```

### Q: 输出视频没有声音
ffmpeg 参数里已经带 `-map 1:a? -c:a aac`，如果原视频有声音应该会自动复制。

---

## 📄 项目结构

```
2Dto3D/
├── mono2stereo_lower_fastinpaint_time_gpu_video.py  # ✅ 主脚本（双模式）
├── mono2stereo_lower_fastinpaint_time_gpu.py        # 单帧版本
├── README.md                                         # ✅ 本文件
├── checkpoints/                                      # 模型权重目录
│   ├── depth_anything_v2_vits.pth
│   └── video_depth_anything_vits.pth
└── submodules/
    ├── depth/dav2/                                   # DepthAnythingV2
    └── Video_Depth_Anything/                         # Video-Depth-Anything
```

---

## 📝 更新日志

### v2.0 (2026-04-23)
- ✅ 集成 Video-Depth-Anything 官方视频模型
- ✅ 流式推理 API，模型内部维护时序状态
- ✅ 单帧模式升级：简单平滑 → 光流对齐 + 平滑
- ✅ 自动保持宽高比，不再强制正方形
- ✅ 完善 --profile-time 性能统计

### v1.5
- ✅ 全 GPU 预处理加速
- ✅ 多线程预读队列
- ✅ NVENC 硬件编码

---

**如果有问题，先加 `--profile-time` 看看哪一步慢，然后再优化！** 😎
