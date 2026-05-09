# 2D 转 3D 立体视频生成

**DepthAnythingV2 + [可选 NVDS 时序稳定] + GPU DIBR + FastInpaint** —— 单目视频转高质量立体视频

> ✅ **三种时序一致性方案可选**：
> 1. 光流后处理平滑
> 2. Video-Depth-Anything 视频模型（内置时序）
> 3. **NVDS 深度稳定器**（新增！）

---

## 🎬 效果预览

| 问题 | 解决前 | 解决后 |
|------|--------|--------|
| **深度抖动** | 帧间深度跳变，闪烁严重 | ✅ 时序一致，平滑过渡 |
| **拉伸变形** | 强制正方形，画面拉伸 | ✅ 保持宽高比，无变形 |
| **运动边缘闪烁** | 物体运动时深度突变 | 🚧 NVDS 优化中 |

---

## 📦 版本对比

| 文件 | 深度模型 | 时序一致性 | 速度 | 推荐度 |
|------|----------|------------|------|--------|
| `mono23d.py` | DepthAnythingV1 单帧 | ❌ 无 | 慢 | ❌ 旧版 |
| `mono2stereo_lower_fastinpaint_time_gpu.py` | DepthAnythingV2 单帧 | ✅ 光流对齐后处理 | 快 | ⭐⭐⭐ |
| `mono2stereo_lower_fastinpaint_time_gpu_video.py` | **Video-Depth-Anything** | ✅✅ 模型级原生时序 | 中 | ⭐⭐⭐⭐⭐ 推荐 |
| **`mono2stereo_with_nvds.py`** | **DAv2/VDA + NVDS 稳定** | ✅✅✅ 三层时序优化 | 较慢 | ⭐⭐⭐ 实验性 |

---

## 🚀 快速开始

### 🆕 方式0：开启 NVDS 时序深度稳定（最新！实验性）

在深度推理后插入 NVDS 时序稳定网络，用当前帧 + 历史 3 帧共同优化当前深度。**可以和视频模型/单帧模型同时开启！**

```bash
python mono2stereo_with_nvds.py \
  --video-path your_video.mp4 \
  --output output_3d_nvds.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --enable-nvds  # ← ✅ 开启 NVDS 时序稳定
```

**可以和视频模型叠加使用（双重时序优化）**：
```bash
python mono2stereo_with_nvds.py \
  --video-path your_video.mp4 \
  --output output_3d_double.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --video-model \   # 视频模型做第一层时序
  --enable-nvds     # NVDS 做第二层时序稳定（双重！）
```

---

### 方式1：视频模型模式（推荐，原生时序一致性）

Video-Depth-Anything 官方流式推理，模型训练时就注入时序监督：

```bash
python mono2stereo_with_nvds.py \
  --video-path your_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --video-model  # ← 开启视频模型
```

### 方式2：单帧 + 光流平滑（无额外权重）

用普通 DepthAnythingV2 单帧模型 + 光流对齐平滑：

```bash
python mono2stereo_with_nvds.py \
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
python mono2stereo_with_nvds.py \
  --video-path your_video.mp4 \
  --output output_3d.mp4 \
  --encoder vits \
  --input-size 518 \
  --max-disparity 16 \
  --fp16 \
  --profile-time  # ← 打印性能分析
```

---

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| **基础参数** | | |
| `--video-path` | 输入视频路径、目录或 txt 列表 | ✅ 必填 |
| `--output` | 输出视频路径（单文件模式） | `./output.mp4` |
| `--input-size` | 深度模型长边像素（必须 14 倍数） | 518 |
| `--encoder` | 模型规模: `vits` / `vitb` / `vitl` | vits |
| `--max-disparity` | 低分辨率下最大视差像素（控制 3D 强度） | 16.0 |
| `--fp16` | 启用 FP16 推理（速度提升 30%+） | ❌ |
| **时序一致性** | | |
| `--video-model` | 使用 Video-Depth-Anything 视频模型（内置时序） | ❌ |
| `--metric` | 视频模型使用 Metric 深度（需要对应权重） | ❌ |
| `--depth-smooth` | 光流时序平滑强度（单帧模式用，0=关闭） | 0.0 |
| `--enable-nvds` | 🆕 **开启 NVDS 时序深度稳定** | ❌ |
| `--nvds-seq-len` | NVDS 参考序列帧数 | 4 |
| `--nvds-ckpt` | NVDS 模型权重路径 | `./submodules/NVDS/NVDS_checkpoints/NVDS_Stabilizer.pth` |
| **输出控制** | | |
| `--layout` | 立体布局: `sbs` 并排 / `ou` 上下 / `overlay` 重合 | sbs |
| `--profile-time` | 输出各阶段耗时统计 | ❌ |

---

## 🧪 NVDS 集成说明（实验性）

### 架构设计
```
输入视频帧
    ↓
[DAv2 单帧 / VDA 视频模型]  # 第一层：原始深度估计
    ↓
[NVDS 时序稳定网络]        # 第二层：利用帧间相关性平滑深度（可选）
    ↓
深度 → 视差转换
    ↓
DIBR 像素重投影 + 补洞
    ↓
3D 视频输出
```

### 当前状态（2026-05-09）
| 状态 | 说明 |
|------|------|
| ✅ 代码跑通 | 可以端到端推理，无 mmcv 依赖 |
| ✅ 模型可以加载 | 权重可以正常载入 |
| ⚠️ 只测试了 6 帧 | 长视频还没测 |
| 🔴 **效果很差** | 深度看起来完全不对，估计和权重/训练有关 |

### 可能的问题
1. **权重不匹配**：原 NVDS 是在 MiDaS/DPT 上训练的，现在换成 DepthAnythingV2，深度分布不一致，直接迁移效果差
2. **输入归一化问题**：DAv2 和原深度模型的输出分布范围不同，NVDS 模型可能需要重新训练
3. **帧填充策略**：前几帧用反向填充，可能影响初期效果

### 下一步计划
- [ ] 对比原 NVDS 官方推理脚本输出，确认深度分布是否一致
- [ ] 在 DAv2 上微调 NVDS 稳定器（需要标注时序视频深度）
- [ ] 对比：光流平滑 vs 视频模型 vs NVDS 三种时序方案的效果差异

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
| VDA 视频模型 | ~12 | 模型自带时序注意力，效果最好 |
| **单帧 + NVDS** | ~8 | 额外加一次 NVDS 推理 |
| **VDA + NVDS** | ~5 | 双重时序稳定，最慢 |

性能瓶颈：
- 🔴 深度推理（最耗时）
- 🟡 NVDS 时序稳定推理
- 🟢 DIBR 像素重投影 + 补洞

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

### NVDS 时序稳定模型
```
submodules/NVDS/NVDS_checkpoints/NVDS_Stabilizer.pth
```
下载地址：https://github.com/baidu-research/NVDS

---

## 🔧 环境依赖

**一个环境就能跑！不需要 mmcv！** 🎉

```bash
pip install torch opencv-python numpy torchvision einops timm matplotlib
```

ffmpeg（需要支持 h264_nvenc）：
```bash
# Ubuntu
sudo apt install ffmpeg
```

### 已移除的依赖
✅ ~~mmcv~~ （完全用纯 PyTorch 重写了 Registry / 权重加载 / 工具函数）  
✅ ~~mmseg~~  
✅ ~~attrs~~ （调试遗留，已删除）  
✅ ~~IPython.embed~~ （调试遗留，已删除）

---

## 🎯 技术亮点

### 1. ✅ 保持宽高比
- 不再强制正方形，自动根据原视频比例计算目标尺寸
- 确保两个维度都是 14 的倍数（DAv2 要求）
- NVDS 自动 pad 到 32 倍数，推理完裁剪回来

### 2. ✅ 三层时序一致性方案可选

| 方案 | 优点 | 缺点 |
|------|------|------|
| **光流对齐后处理** | 无额外权重，速度快，即插即用 | 大运动场景光流不准 |
| **Video-Depth-Anything** | 原生级一致性，运动场景也不糊 | 需要额外视频模型权重 |
| **NVDS 时序稳定** | 专门为深度时序稳定设计 | 目前效果差，需要重新训练 |

### 3. ✅ 全 GPU 加速
- 预处理（Resize + Normalize）全部在 GPU
- BGR→RGB 颜色转换 GPU 加速
- DIBR 像素重投影 GPU 并行
- FastInpaint 补洞 GPU 卷积
- 多线程预读队列，CPU-GPU 流水线并行

---

## 🐛 常见问题

### Q: 报错 `ModuleNotFoundError: No module named '...'`
```bash
pip install einops timm matplotlib
```

### Q: NVDS 深度效果完全不对
已知问题！目前直接把在 MiDaS 上训练的 NVDS 权重迁移到 DAv2，深度分布不一致，效果很差。需要：
1. 要么微调 NVDS 权重适配 DAv2
2. 要么把 DAv2 深度归一化到和 MiDaS 一样的分布

### Q: 3D 效果太强/太弱
调 `--max-disparity`：
- 减小 → 3D 更浅
- 增大 → 3D 更强

### Q: 左右眼反了
简单，把 `compose_stereo` 里的顺序换一下：
```python
return np.concatenate([right_u8, left_u8], axis=1)
```

---

## 📄 项目结构

```
2Dto3D/
├── mono2stereo_with_nvds.py                    # ✅ 最新主脚本（支持所有模式）
├── mono2stereo_lower_fastinpaint_time_gpu_video.py  # 旧版（双模式）
├── README.md                                    # ✅ 本文件
├── checkpoints/                                 # 模型权重目录
│   ├── depth_anything_v2_vits.pth
│   └── video_depth_anything_vits.pth
└── submodules/
    ├── depth/dav2/                              # DepthAnythingV2
    ├── Video_Depth_Anything/                    # Video-Depth-Anything
    └── NVDS/                                     # 🆕 NVDS 时序稳定器
        ├── NVDS_checkpoints/NVDS_Stabilizer.pth
        ├── full_model.py
        └── ...
```

---

## 📝 更新日志

### v2.1 (2026-05-09)
- ✅ **新增 NVDS 时序深度稳定集成**
- ✅ `--enable-nvds` 开关，默认关闭（零性能影响）
- ✅ 支持与 `--video-model` 同时开启（双重时序优化）
- ✅ 彻底移除 NVDS 的 mmcv / mmseg 依赖（纯 PyTorch）
- ✅ 自动分辨率适配：14 倍数 → 32 倍数 → 裁剪回原尺寸
- ⚠️ 目前效果较差，需要进一步调优/微调

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

**NVDS 目前效果还有问题，欢迎一起调优！先把视频模型用起来效果就很好了！** 😎
