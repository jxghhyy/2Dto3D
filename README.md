# 2D 转 3D 立体视频生成

**DepthAnythingV2 / Video-Depth-Anything + Plan A-E 五重时序稳定 + GPU DIBR + FastInpaint** —— 单目视频转高质量立体视频

> ✅ **两种方案可选：
> 1. 单帧模型 + Plan A-E 五重时序稳定（推荐，无需额外权重）
> 2. Video-Depth-Anything 视频模型（官方流式推理）

---

## 🎬 效果预览

| 问题 | 解决前 | 解决后 |
|------|--------|--------|
| **深度抖动** | 帧间深度跳变，闪烁严重 | ✅ 五重时序稳定，丝滑过渡 |
| **拉伸变形** | 强制正方形，画面拉伸 | ✅ 保持宽高比，无变形 |
| **前景边缘** | 被背景侵蚀 | ✅ 空洞左侧膨胀保护 |

---

## 📦 版本对比

| 文件 | 深度模型 | 时序一致性 | 速度 | 推荐度 |
|------|----------|------------|------|--------|
| `mono2stereo_video_better_111.py | DepthAnythingV2 / Video-Depth-Anything | ✅✅✅✅✅ Plan A-E 五重时序稳定 | 快 | ⭐⭐⭐⭐⭐ **推荐** |

---

## 🚀 快速开始

### 方式1：单帧模型 + Plan A-E 五重时序稳定（推荐）

```bash
python mono2stereo_video_better_111.py \
  --video-path test_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --dibr-size 1080 \
  --max-disparity 20 \
  --fp16 \
  --depth-smooth 0.6 \
  --quantile-smooth 0.8 \
  --rgb-motion-sigma 0.1 \
  --flow-align \
  --flow-height 144 \
  --median-window 3
```

**Plan A-E 参数说明**：
- `--depth-smooth 0.6` (Plan B+C): near 空间 EMA 强度（0~1，越大越平滑）
- `--quantile-smooth 0.8` (Plan A): 分位数 EMA 强度（消除整帧尺度漂移）
- `--rgb-motion-sigma 0.1` (Plan B): RGB 运动敏感度（越小越敏感）
- `--flow-align` (Plan D): 启用光流对齐 Motion-aligned EMA
- `--flow-height 144`: 光流计算分辨率高度（144 推荐，速度精度平衡）
- `--median-window 3` (Plan E): 滑窗中值窗口大小（3 推荐）

---

### 方式2：Video-Depth-Anything 视频模型

```bash
python mono2stereo_video_better_111.py \
  --video-path test_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 20 \
  --fp16 \
  --video-model
```

---

## ⚙️ 参数说明

### 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video-path` | 输入视频路径/目录/txt | ✅ 必填 |
| `--output` | 输出视频路径（单文件） | `./output.mp4` |
| `--input-size` | 深度模型分辨率长边（必须 14 倍数） | 518 |
| `--dibr-size` | DIBR 渲染分辨率高度（-1=原分辨率） | -1 |
| `--encoder` | 模型规模: `vits`/`vitb`/`vitl` | vits |
| `--max-disparity` | 原分辨率最大视差像素（3D 强度） | 16.0 |
| `--fp16` | 启用 FP16 推理（快 30%+） | ❌ |

### Plan A-E 时序稳定（仅单帧模式）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--depth-smooth` | Plan B+C: near 空间 EMA 强度 | 0.6 |
| `--quantile-smooth` | Plan A: 分位数 EMA 强度 | 0.8 |
| `--rgb-motion-sigma` | Plan B: RGB 运动敏感度 | 0.1 |
| `--flow-align` | Plan D: 启用光流对齐 Motion-aligned EMA | ❌ |
| `--flow-height` | Plan D: 光流计算分辨率高度 | 144 |
| `--median-window` | Plan E: 滑窗中值窗口大小（1/3/5/7） | 1 |

### DIBR & 补洞

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hole-dilate-left` | 空洞左侧膨胀像素（保护前景边缘） | 0 |
| `--fast-kernel` | FastInpaint 邻域核大小 | 7 |
| `--fast-max-iter` | FastInpaint 最大迭代次数 | 64 |

### 输出 & 编码

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--layout` | 立体布局: `sbs` 并排 / `ou` 上下 / `anaglyph` 红青 / `overlay` 重合 | sbs |
| `--nvenc-preset` | NVENC 编码预设 | p4 |
| `--nvenc-cq` | NVENC CQ 质量（越小质量越高） | 19 |
| `--profile-time` | 打印各阶段耗时统计 | ❌ |

---

## 🎯 Plan A-E 五重时序稳定详解

```
原始深度 → [Plan A: 分位数 EMA] → [Plan C: near 空间] → [Plan D: 光流对齐] → [Plan B: RGB 引导自适应 EMA] → [Plan E: 滑窗中值] → 稳定深度
```

| 方案 | 作用 | 耗时 |
|------|------|------|
| **Plan A** | 消除整帧尺度/偏移漂移（随机采样 16k 像素，几乎零开销） | ~0.1ms |
| **Plan B** | RGB 引导逐像素自适应平滑（静止区强平滑，运动区不平滑，无 ghosting） | ~0.5ms |
| **Plan C** | 平滑在归一化后的 near 空间（与渲染量直接相关） | ~0.05ms |
| **Plan D** | 光流对齐 Motion-aligned EMA（低分辨率 Farneback） | ~1~3ms |
| **Plan E** | 滑窗逐像素中值（抗单帧极值噪声） | ~0.3ms |

---

## 📊 性能对比（3090）

以 1920×1080 视频为例：

| 模式 | FPS | 说明 |
|------|-----|------|
| 单帧模型 + Plan A-E（input=518, dibr=1080) | ~25 FPS | 推荐配置 |
| 单帧模型 + Plan A-E（input=518, dibr=-1) | ~18 FPS | 全分辨率渲染 |
| Video-Depth-Anything 视频模型 | ~12 FPS | 官方流式推理 |

性能瓶颈：
- 🔴 深度推理（最耗时）
- 🟡 Plan D 光流对齐（可选）
- 🟢 DIBR + 补洞

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
checkpoints/metric_video_depth_anything_vits.pth
```
下载地址：https://github.com/DepthAnything/Video-Depth-Anything

---

## 🔧 环境依赖

```bash
pip install torch opencv-python numpy torchvision einops timm matplotlib
```

ffmpeg（需要支持 h264_nvenc）：
```bash
# Ubuntu
sudo apt install ffmpeg
```

---

## 💡 3D 强度调整

主要调 `--max-disparity` 参数：

| 值 | 效果 | 适用场景 |
|----|------|----------|
| 8~12 | 浅 3D，出屏效果弱 | 纪录片、风景 |
| **16~20** | 平衡，推荐 | 一般视频 |
| 20~30 | 强 3D，出屏明显 | 人物特写、动画 |

> 提示：不要超过 40，否则会有明显伪影

---

## 🎯 技术亮点

### 1. ✅ Plan A-E 五重时序稳定
- 全在深度推理分辨率完成，几乎不影响速度
- Plan A 分位数 EMA 消除整帧漂移
- Plan B RGB 引导避免 ghosting
- Plan D 光流对齐 Motion-aligned
- Plan E 中值滤波抗噪

### 2. ✅ 双分辨率 Pipeline
- 深度推理：低分辨率（快）
- DIBR 渲染：高分辨率（质量好）
- 一次 GPU 上传，同时生成双分辨率

### 3. ✅ int64 无损 Z-buffer
- 彻底解决像素冲突伪影
- 前景永远在背景前面

### 4. ✅ 全 GPU 加速
- 预处理（Resize + Normalize）全 GPU
- BGR→RGB 颜色转换 GPU 加速
- DIBR 像素重投影 GPU 并行
- FastInpaint 补洞 GPU 卷积
- 多线程预读队列，CPU-GPU 流水线并行

---

## 🐛 常见问题

### Q: 3D 效果太强/太弱
调 `--max-disparity`：
- 减小 → 3D 更浅
- 增大 → 3D 更强

### Q: 左右眼反了
简单，把 `compose_stereo` 里的顺序换一下：
```python
return np.concatenate([right_u8, left_u8], axis=1)
```

### Q: 前景边缘被背景侵蚀
增加 `--hole-dilate-left` 参数：
- 比如 `--hole-dilate-left 3` 或 `5`

### Q: 闪烁还是比较严重
增强时序稳定：
- 增加 `--depth-smooth` 到 0.7~0.8
- 增加 `--quantile-smooth` 到 0.9
- 启用 `--flow-align`
- 设置 `--median-window 3`

---

## 📝 更新日志

### v3.0（最新）
- ✅ **Plan A-E 五重时序稳定**（替代 NVDS）
- ✅ **深度推理 & DIBR 渲染分辨率彻底解耦**（--dibr-size）
- ✅ 性能优化：inpaint kernel 缓存、随机 16k 像素 quantile
- ✅ 全 GPU 双分辨率预处理 pipeline
- ✅ 移除 NVDS 相关代码（效果不佳且需要重训）

### v2.0
- ✅ 集成 Video-Depth-Anything 官方视频模型
- ✅ 保持宽高比

### v1.5
- ✅ 全 GPU 预处理加速
- ✅ 多线程预读队列
- ✅ NVENC 硬件编码

---

**推荐用单帧模型 + Plan A-E 五重时序稳定，无需额外权重，效果好速度快！** 😎

