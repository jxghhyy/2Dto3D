# DIBR 空洞修复方法对比

根据图像修复.md的建议，实现了三种不同的 inpainting 方法，用于 DIBR 空洞修复对比测试。

## 方法一览

| 方法 | 文件 | 核心思想 | 适合场景 | 理论基础 |
|------|------|----------|----------|----------|
| **方法1: 水平背景拉伸** | `mono2stereo_horizontal_background_pull.py` | 同一行背景像素加权，优先从背景侧拉伸 | DIBR 专用，速度极快 | 观察到 DIBR 空洞通常沿垂直边缘出现 |
| **方法2: 方向壳填充** | `mono2stereo_directional_shell.py` | 边界向内部逐层推进，多方向采样，深度加权 | DIBR 专用，质量较好 | Guidefill + Coherence Transport |
| **方法3: Telea Fast Marching** | `mono2stereo_telea_fast_marching.py` | 经典传统 inpaint，GPU近似实现 | 通用 inpaint 基准对比 | Telea 2004 论文 |

## 测试命令

### 方法1: 水平背景拉伸
```bash
python mono2stereo_horizontal_background_pull.py \
  --video-path /mnt/A/jiangxg/dataset/shuai/Huawei/CL_12s.mp4 \
  --outdir ./output_hbp \
  --input-size 518 \
  --encoder vits \
  --hbp-max-distance 32 \
  --hbp-sigma 8.0 \
  --bg-threshold 0.3 \
  --profile-time
```

**可调参数:**
- `--hbp-max-distance`: 每行搜索最大距离（默认32）
- `--hbp-sigma`: 距离权重衰减系数（默认8.0）
- `--bg-threshold`: 背景深度阈值（默认0.3）

---

### 方法2: 方向壳填充
```bash
python mono2stereo_directional_shell.py \
  --video-path /mnt/A/jiangxg/dataset/shuai/Huawei/CL_12s.mp4 \
  --outdir ./output_ds \
  --input-size 518 \
  --encoder vits \
  --ds-max-samples 8 \
  --ds-sample-distance 3 \
  --ds-sigma-dist 8.0 \
  --ds-sigma-depth 0.2 \
  --ds-max-iter 32 \
  --bg-threshold 0.3 \
  --profile-time
```

**可调参数:**
- `--ds-max-samples`: 每个方向最大采样点数（默认8）
- `--ds-sample-distance`: 采样点间距（默认3）
- `--ds-sigma-dist`: 距离权重衰减（默认8.0）
- `--ds-sigma-depth`: 深度权重衰减（默认0.2）
- `--ds-max-iter`: 最大迭代次数（默认32）
- `--ds-enable-diagonal`: 启用斜向采样（默认开启）

---

### 方法3: Telea Fast Marching
```bash
python mono2stereo_telea_fast_marching.py \
  --video-path /mnt/A/jiangxg/dataset/shuai/Huawei/CL_12s.mp4 \
  --outdir ./output_telea \
  --input-size 518 \
  --encoder vits \
  --telea-radius 5 \
  --telea-sigma 1.5 \
  --telea-max-iter 64 \
  --bg-threshold 0.3 \
  --telea-use-depth \
  --profile-time
```

**可调参数:**
- `--telea-radius`: 采样半径（推荐3-5）
- `--telea-sigma`: 距离衰减系数（推荐1.0-2.0）
- `--telea-max-iter`: 最大迭代次数（默认64）
- `--telea-use-depth`: 启用深度感知（只从背景采样）

---

## 预期效果对比

### 方法1: 水平背景拉伸
- ✅ 速度最快（O(n) 线性时间）
- ✅ 无前景颜色污染（纯背景采样）
- ⚠️ 对斜线、栏杆等复杂背景效果一般
- ⚠️ 适合作为第一阶段填充

### 方法2: 方向壳填充
- ✅ 保留结构信息最好
- ✅ 多方向采样适应性强
- ✅ 背景方向自动判断
- ⚠️ 速度中等（迭代式）
- 💡 推荐作为主力方法

### 方法3: Telea Fast Marching
- ✅ 经典算法，理论完善
- ⚠️ GPU近似实现，比原版CPU快
- ⚠️ 不区分前景/背景时，容易出现前景污染
- 💡 作为对比基准，看 DIBR 专用方法的优势

---

## 共同参数

所有三个脚本都支持以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-size` | 深度推理长边像素 | 518 |
| `--dibr-size` | DIBR 渲染高度，-1=原分辨率，0=同深度 | -1 |
| `--encoder` | 深度模型: vits/vitb/vitl | vits |
| `--max-disparity` | 最大视差像素（原分辨率） | 24.0 |
| `--hole-dilate-left` | 空洞向左膨胀像素 | 0 |
| `--hole-dilate-right` | 空洞向右膨胀像素 | 1 |
| `--layout` | 输出格式: sbs/ou/anaglyph | sbs |
| `--profile-time` | 打印各阶段用时统计 | 关 |

---

## 评估标准

对比三个方法输出时，重点观察：

1. **前景边缘质量**: 是否有前景颜色"渗"到背景空洞中
2. **结构延续性**: 横线、竖线、斜线是否自然延续
3. **背景纹理**: 草地、墙面等纹理是否平滑
4. **速度**: FPS、单帧耗时
5. **光晕/伪影**: 是否有异常的颜色块、模糊区域

---

## 推荐流程

1. 先用**方法1**做快速测试，验证深度和DIBR流程没问题
2. 再用**方法2**做主力，调节参数找到最佳效果
3. 最后用**方法3**做对比，看 DIBR 专用方法的优势

如果时间充足，可以测试不同视频（人像、风景、建筑、运动），观察各方法的适应性差异。
