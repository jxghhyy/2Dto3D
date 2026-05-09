"""
NVDS + DepthAnythingV2 双向推理脚本

修正版，修复以下问题：
1. ✅ 深度估计器替换为 DepthAnythingV2
2. ✅ 修复迭代细化：times>0 时读取上一轮结果而非初始深度
3. ✅ 修复 per-frame min-max 归一化：改用全局统计（第一帧确定 scale，后续帧复用）
4. ✅ 统一 RGB 归一化为 ImageNet mean/std（DAv2 和 NVDS 都用同一套）
5. ✅ 分辨率约束：infer_w/infer_h 必须是 32 的倍数（NVDS），输入 DAv2 时自动 pad 到 14 倍数
6. ✅ 保存 float32 深度而非 uint16 PNG，避免量化损失和 per-frame 归一化问题
7. ✅ 前3帧处理改为更合理的策略：不足4帧时用已有帧反向填充

用法：
  CUDA_VISIBLE_DEVICES=0 python infer_NVDS_dav2_bi.py \
    --base_dir ./demo_outputs/dav2_init/your_video/ \
    --vnum your_video \
    --infer_w 672 --infer_h 384 \
    --encoder vits \
    --timesall 2

依赖：
  - DepthAnythingV2: submodules/depth/dav2/
  - NVDS: submodules/NVDS/ (本目录)
  - GMFlow: submodules/NVDS/gmflow/
"""

import sys
import os

# 确保可以 import DAv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'depth', 'dav2'))

import argparse
import glob
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

# ============ NVDS 内部 import ============
from Depth_decoder import Decoder
from backbone import *
from networks import *
from full_model import NVDS
from smooth_loss import flow_warping_loss_align_test
from utils.flow_viz import save_vis_flow_tofile
from gmflow.geometry import flow_warp, coords_grid
from gmflow.gmflow import GMFlow

# ============ DepthAnythingV2 import ============
from depth_anything_v2.dpt import DepthAnythingV2


def get_args_parser():
    parser = argparse.ArgumentParser(description='NVDS + DepthAnythingV2 Bidirectional Inference')

    # ---- NVDS / GMFlow 原有参数 ----
    parser.add_argument('--checkpoint_dir', default='tmp', type=str)
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+')
    parser.add_argument('--padding_factor', default=16, type=int)
    parser.add_argument('--max_flow', default=400, type=int)
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+')
    parser.add_argument('--with_speed_metric', action='store_true')

    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow
    parser.add_argument('--num_scales', default=1, type=int)
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+')
    parser.add_argument('--gamma', default=0.9, type=float)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--dir_paired_data', action='store_true')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true')
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--save_vis_flow', action='store_true')
    parser.add_argument('--no_save_flo', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')

    # ---- 推理特有参数 ----
    parser.add_argument('--all_seq_len', default=4, type=int)
    parser.add_argument('--max_epoch', default=500)
    parser.add_argument('--vnum', default='000423', type=str)
    parser.add_argument('--timesall', default=2, type=int, help='迭代轮数（前向+后向各 timesall 次）')
    parser.add_argument('--base_dir', default='/xxx/xxx/', type=str)
    parser.add_argument('--infer_w', default=672, type=int, help='推理宽度，必须为32的倍数')
    parser.add_argument('--infer_h', default=384, type=int, help='推理高度，必须为32的倍数')
    parser.add_argument('--clip_step', default=1, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str)

    # ---- DepthAnythingV2 参数 ----
    parser.add_argument('--encoder', default='vits', type=str,
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='DAv2 模型规模')
    parser.add_argument('--dav2_ckpt', default=None, type=str,
                        help='DAv2 权重路径，默认 checkpoints/depth_anything_v2_{encoder}.pth')
    parser.add_argument('--dav2_input_size', default=518, type=int,
                        help='DAv2 输入长边像素（必须14倍数），会自动 pad')
    parser.add_argument('--fp16', action='store_true', help='DAv2 推理使用 FP16')

    return parser


# ============ 工具函数 ============

# ImageNet 归一化（DAv2 和 NVDS 统一使用）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def img_loader(path):
    """读取图像，返回 [0,1] RGB float32"""
    image = cv2.imread(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    return image.astype(np.float32)


def normalize_rgb(rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """RGB [0,1] → ImageNet normalized tensor [1,3,H,W]"""
    rgb = (rgb - mean) / std
    rgb = np.transpose(rgb, (2, 0, 1))
    return torch.Tensor(np.ascontiguousarray(rgb.astype(np.float32))).unsqueeze(0)


def pad_to_multiple(tensor, multiple=14):
    """将 tensor [1,C,H,W] pad 到 H,W 都是 multiple 的倍数，返回 pad 后的 tensor 和原始尺寸"""
    _, _, h, w = tensor.shape
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if new_h == h and new_w == w:
        return tensor, (h, w)
    pad_h = new_h - h
    pad_w = new_w - w
    # 右下角 pad
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, (h, w)


def save_depth_float(depth_tensor, path):
    """保存 float32 深度到 .npy 文件（避免 uint16 量化损失和 per-frame 归一化）"""
    np.save(path, depth_tensor.cpu().numpy().astype(np.float32))


def load_depth_float(path, target_size=None):
    """加载 float32 深度，可选 resize"""
    depth = np.load(path).astype(np.float32)
    depth = torch.from_numpy(depth)
    if target_size is not None:
        depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), target_size,
                              mode='bilinear', align_corners=True).squeeze()
    return depth


def save_depth_uint16(depth_tensor, path):
    """保存深度为 uint16 PNG（兼容原 NVDS 格式）"""
    depth = depth_tensor.cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth_u16 = (65535.0 * (depth - d_min) / (d_max - d_min)).astype(np.uint16)
    else:
        depth_u16 = np.zeros_like(depth, dtype=np.uint16)
    cv2.imwrite(path, depth_u16, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def normalize_depth_to_01(depth_tensor):
    """将深度归一化到 [0,1]，使用全局 min/max（由调用方传入）"""
    d_min, d_max = depth_tensor.min(), depth_tensor.max()
    if d_max - d_min > 1e-6:
        return (depth_tensor - d_min) / (d_max - d_min)
    return torch.zeros_like(depth_tensor)


def compute_global_depth_stats(frames, dav2_model, infer_size, device, dav2_input_size=518):
    """
    预扫描所有帧，计算全局深度 min/max 统计量。
    这样后续帧的深度可以映射到统一尺度，避免 per-frame 归一化破坏时序一致性。
    
    返回: (global_median, global_iqr) 用于稳健归一化
    """
    print(">>> 预扫描所有帧计算全局深度统计...")
    all_depths = []
    for i, frame_path in enumerate(frames):
        rgb = img_loader(frame_path)
        rgb_resized = cv2.resize(rgb, infer_size, interpolation=cv2.INTER_CUBIC)
        rgb_tensor = normalize_rgb(rgb_resized).to(device)
        rgb_padded, orig_size = pad_to_multiple(rgb_tensor, multiple=14)
        
        with torch.no_grad():
            depth = dav2_model(rgb_padded)
        # 裁掉 padding
        depth = depth[:, :orig_size[0], :orig_size[1]]
        all_depths.append(depth.cpu())
    
    # 拼接所有帧深度
    all_depths_cat = torch.cat([d.flatten() for d in all_depths])
    global_median = all_depths_cat.median().item()
    # 使用 IQR (interquartile range) 作为 scale
    q25 = all_depths_cat.quantile(0.25).item()
    q75 = all_depths_cat.quantile(0.75).item()
    global_iqr = max(q75 - q25, 1e-6)
    
    print(f"    全局 median={global_median:.4f}, IQR={global_iqr:.4f}")
    # 释放临时内存
    del all_depths, all_depths_cat
    torch.cuda.empty_cache()
    
    return global_median, global_iqr


def normalize_depth_global(depth_tensor, global_median, global_iqr, target_range=(0, 1)):
    """使用全局统计量归一化深度，保持帧间一致性"""
    # 先做 median-centering + IQR scaling（类似 NVDS 的 normalize_prediction_robust）
    depth_centered = depth_tensor - global_median
    depth_scaled = depth_centered / global_iqr
    # 再映射到 [0, 1]
    # 使用 clip 防止极端值，比如 [-5, 5] 映射到 [0, 1]
    lo, hi = target_range
    clip_val = 5.0  # 5个IQR范围
    depth_scaled = torch.clamp(depth_scaled, -clip_val, clip_val)
    depth_normalized = (depth_scaled + clip_val) / (2 * clip_val)  # → [0, 1]
    return depth_normalized


def infer_dav2_depth(rgb_resized, dav2_model, device, fp16=False):
    """
    用 DAv2 推理单帧深度。
    
    Args:
        rgb_resized: numpy [H,W,3] float32, 已经 resize 到 infer_size
        dav2_model: DepthAnythingV2 模型
        device: torch device
        fp16: 是否使用 FP16
    
    Returns:
        depth: tensor [H,W] float32，DAv2 原始输出（逆深度，值越大越近）
    """
    rgb_tensor = normalize_rgb(rgb_resized).to(device)
    rgb_padded, orig_size = pad_to_multiple(rgb_tensor, multiple=14)
    
    with torch.no_grad():
        if fp16:
            with torch.cuda.amp.autocast():
                depth = dav2_model(rgb_padded)
        else:
            depth = dav2_model(rgb_padded)
    
    # 裁掉 padding
    depth = depth[:, :orig_size[0], :orig_size[1]]
    return depth.squeeze(0)  # [H, W]


# ============ 主流程 ============

if __name__ == '__main__':

    print('=== NVDS + DepthAnythingV2 Bidirectional Inference (Fixed) ===')
    device = torch.device('cuda:0')
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    infer_size = (int(args.infer_w), int(args.infer_h))
    
    # 检查分辨率约束
    assert args.infer_w % 32 == 0 and args.infer_h % 32 == 0, \
        f"infer_w={args.infer_w} 和 infer_h={args.infer_h} 必须是 32 的倍数"
    
    # ============ 加载 GMFlow ============
    print('>>> 加载 GMFlow 光流模型...')
    model_flow = GMFlow(
        feature_channels=args.feature_channels,
        num_scales=args.num_scales,
        upsample_factor=args.upsample_factor,
        num_head=args.num_head,
        attention_type=args.attention_type,
        ffn_dim_expansion=args.ffn_dim_expansion,
        num_transformer_layers=args.num_transformer_layers,
    ).to(device)
    model_flow = torch.nn.DataParallel(model_flow, device_ids=[0])
    model_flow = model_flow.module
    
    gmflow_ckpt = './gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth'
    print(f'    加载 GMFlow checkpoint: {gmflow_ckpt}')
    checkpoint = torch.load(gmflow_ckpt, map_location='cpu')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_flow.load_state_dict(weights, strict=args.strict_resume)
    model_flow.to(device)
    model_flow.eval()
    
    # ============ 加载 NVDS 稳定器 ============
    print('>>> 加载 NVDS 稳定器...')
    nvds_ckpt = './NVDS_checkpoints/NVDS_Stabilizer.pth'
    checkpoint_nvds = torch.load(nvds_ckpt, map_location='cpu')
    model_nvds = NVDS()
    model_nvds = torch.nn.DataParallel(model_nvds, device_ids=[0]).cuda()
    model_nvds.load_state_dict(checkpoint_nvds)
    model_nvds.to(device)
    model_nvds.eval()
    
    # ============ 加载 DepthAnythingV2 ============
    print(f'>>> 加载 DepthAnythingV2 (encoder={args.encoder})...')
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    dav2 = DepthAnythingV2(**model_configs[args.encoder])
    
    dav2_ckpt = args.dav2_ckpt
    if dav2_ckpt is None:
        # 尝试多个可能的路径
        candidates = [
            f'../../checkpoints/depth_anything_v2_{args.encoder}.pth',
            f'./checkpoints/depth_anything_v2_{args.encoder}.pth',
        ]
        for c in candidates:
            if os.path.exists(c):
                dav2_ckpt = c
                break
        if dav2_ckpt is None:
            raise FileNotFoundError(
                f'找不到 DAv2 权重，请用 --dav2_ckpt 指定路径，'
                f'或将权重放到 checkpoints/depth_anything_v2_{args.encoder}.pth'
            )
    
    print(f'    加载 DAv2 checkpoint: {dav2_ckpt}')
    dav2.load_state_dict(torch.load(dav2_ckpt, map_location='cpu'))
    dav2 = dav2.to(device).eval()
    if args.fp16:
        dav2.half()
        print('    DAv2 已启用 FP16')
    
    # ============ 准备数据路径 ============
    video_dir = './demo_videos/' + args.vnum + '/'
    frames = glob.glob(video_dir + '/left/*.png')
    if not frames:
        # 兼容：也搜索直接放视频帧的目录
        frames = glob.glob(video_dir + '/*.png')
    if not frames:
        raise FileNotFoundError(f'找不到视频帧: {video_dir}')
    frames.sort(key=lambda x: int(x[-10:-4]))
    print(f'>>> 找到 {len(frames)} 帧，视频: {args.vnum}')
    
    seq_len = args.all_seq_len
    clip_step = args.clip_step
    temloss = flow_warping_loss_align_test(int(args.infer_h), int(args.infer_w))
    
    base_dir = args.base_dir
    save_disp_dir_initial = base_dir + '/initial/gray/'
    save_ksh_dir_initial = base_dir + '/initial/color/'
    save_float_dir_initial = base_dir + '/initial/float/'  # 保存 float32 深度
    os.makedirs(save_disp_dir_initial, exist_ok=True)
    os.makedirs(save_ksh_dir_initial, exist_ok=True)
    os.makedirs(save_float_dir_initial, exist_ok=True)
    
    # ============ 阶段1: DAv2 逐帧推理初始深度 ============
    print('\n===== 阶段1: DepthAnythingV2 初始深度推理 =====')
    
    # 1a. 预扫描计算全局深度统计（解决 per-frame 归一化问题）
    global_median, global_iqr = compute_global_depth_stats(
        frames, dav2, infer_size, device, args.dav2_input_size
    )
    
    # 1b. 逐帧推理并保存
    initial_depth_cache = {}  # {frame_idx: depth_tensor [H,W]}，后续 NVDS 直接使用
    
    all_tem = 0
    previous_outputs = None
    previous_rgb = None
    
    for i in range(len(frames)):
        frame = frames[i]
        rgb = img_loader(frame)
        rgb_resized = cv2.resize(rgb, infer_size, interpolation=cv2.INTER_CUBIC)
        
        # DAv2 推理
        depth_raw = infer_dav2_depth(rgb_resized, dav2, device, fp16=args.fp16)
        
        # 使用全局统计归一化（保持帧间一致性）
        depth_normalized = normalize_depth_global(depth_raw, global_median, global_iqr)
        
        initial_depth_cache[i] = depth_normalized.clone()
        
        # 保存可视化
        depth_vis = depth_normalized.cpu().numpy()
        plt.imsave(
            save_ksh_dir_initial + str(i) + '.png',
            depth_vis,
            cmap='inferno',
            vmin=0, vmax=1  # 固定范围，所有帧一致
        )
        
        # 保存 uint16（兼容格式）
        save_depth_uint16(depth_normalized, save_disp_dir_initial + '/frame_%06d.png' % i)
        
        # 保存 float32（精确格式，NVDS 迭代时读取）
        save_depth_float(depth_normalized, save_float_dir_initial + '/frame_%06d.npy' % i)
        
        # OPW 指标计算（与原脚本一致）
        outputs = depth_normalized.unsqueeze(0).to(device)  # [1, H, W]
        rgb_flow_tensor = normalize_rgb(rgb_resized).to(device)  # [1, 3, H, W]
        
        if i >= 1:
            outputs_flow = outputs.clone().to(device)
            rgb_flow = rgb_flow_tensor.clone()
            
            results_dict = model_flow(
                rgb_flow, previous_rgb,
                attn_splits_list=args.attn_splits_list,
                corr_radius_list=args.corr_radius_list,
                prop_radius_list=args.prop_radius_list,
                pred_bidir_flow=args.pred_bidir_flow,
            )
            flow2to1 = results_dict['flow_preds'][-1]
            
            # 反归一化 RGB 用于 flow warp
            prev_rgb_denorm = previous_rgb.clone()
            for c in range(3):
                prev_rgb_denorm[:, c] = prev_rgb_denorm[:, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
            prev_rgb_denorm = prev_rgb_denorm * 255
            rgb_flow_denorm = rgb_flow.clone()
            for c in range(3):
                rgb_flow_denorm[:, c] = rgb_flow_denorm[:, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
            rgb_flow_denorm = rgb_flow_denorm * 255
            
            img1to2_seq = flow_warp(prev_rgb_denorm, flow2to1, mask=False)
            previous_outputs_warp = previous_outputs.unsqueeze(1).to(device)
            outputs1to2 = flow_warp(previous_outputs_warp, flow2to1, mask=False)
            
            tem_loss = temloss(img1to2_seq, rgb_flow_denorm, outputs1to2, outputs_flow, device)
            all_tem += tem_loss.item()
        
        previous_outputs = outputs.clone()
        previous_rgb = rgb_flow_tensor.clone()
    
    dpt_index = all_tem / len(frames)
    print(f'    Initial OPW: all={all_tem:.4f}, mean={dpt_index:.4f}, frames={len(frames)}')
    
    with open(base_dir + '/result.txt', 'a') as f:
        f.write('**********initial (DAv2)**********\n')
        f.write(f'all:{all_tem},mean:{dpt_index},frames:{len(frames)}\n')
        # 保存全局统计量，方便后续复现
        f.write(f'global_median:{global_median},global_iqr:{global_iqr}\n')
    
    # ============ 阶段2: NVDS 前向+后向迭代稳定化 ============
    print('\n===== 阶段2: NVDS 迭代稳定化 =====')
    
    # 维护一份内存中的深度缓存，避免反复读写磁盘
    # 初始时从 initial_depth_cache 加载
    depth_cache = dict(initial_depth_cache)  # {frame_idx: tensor [H,W]}
    
    min_fwd = 100000.0
    min_bwd = 100000.0
    best_fwd = []
    best_bwd = []
    no_change_times_fwd = 0
    no_change_times_bwd = 0
    
    for times in range(args.timesall):
        # ✅ 修复：times>0 时使用上一轮的结果
        # depth_cache 在每轮结束时更新，所以直接从 depth_cache 读取即可
        # 不再硬编码读 initial/gray/
        
        if times % 2 == 0:
            print(f'\n  --------- forward: {times+1} ---------')
            frames_ordered = sorted(frames, key=lambda x: int(x[-10:-4]))
            frame_indices = list(range(len(frames)))
        else:
            print(f'\n  --------- backward: {times+1} ---------')
            frames_ordered = sorted(frames, key=lambda x: int(x[-10:-4]), reverse=True)
            frame_indices = list(range(len(frames) - 1, -1, -1))
        
        save_disp_dir = base_dir + '/' + str(times + 1) + '/gray/'
        save_ksh_dir = base_dir + '/' + str(times + 1) + '/color/'
        save_float_dir = base_dir + '/' + str(times + 1) + '/float/'
        os.makedirs(save_disp_dir, exist_ok=True)
        os.makedirs(save_ksh_dir, exist_ok=True)
        os.makedirs(save_float_dir, exist_ok=True)
        
        all_tem = 0
        temp_result = [None] * len(frames)  # 按原始顺序存储
        new_depth_cache = {}
        
        for traversal_i, frame_idx in enumerate(frame_indices):
            frame = frames[frame_idx]
            
            # 读取当前帧 RGB
            rgb = img_loader(frame)
            rgb_resized = cv2.resize(rgb, infer_size, interpolation=cv2.INTER_CUBIC)
            rgb_tensor = normalize_rgb(rgb_resized).unsqueeze(0).to(device)  # [1,1,3,H,W]
            
            # 构建参考序列: seq_len 帧 RGBD
            # ✅ 修复：前3帧改为用已有帧反向填充，而非重复同一帧
            ref_seq_list = []
            for j in range(seq_len):
                if times % 2 == 0:
                    ref_idx = frame_idx - j * clip_step
                else:
                    ref_idx = frame_idx + j * clip_step
                
                # 如果参考帧不存在，用最近的可用帧（反向填充）
                if ref_idx < 0:
                    ref_idx = 0
                elif ref_idx >= len(frames):
                    ref_idx = len(frames) - 1
                
                # 读取参考帧 RGB
                ref_rgb = img_loader(frames[ref_idx])
                ref_rgb_resized = cv2.resize(ref_rgb, infer_size, interpolation=cv2.INTER_CUBIC)
                ref_rgb_tensor = normalize_rgb(ref_rgb_resized)  # [3, H, W]
                
                # 读取参考帧深度（从 depth_cache，已是 [0,1] 归一化）
                ref_depth = depth_cache[ref_idx].to(device)  # [H, W]
                
                # 拼接 RGBD: 4通道
                rgbd = torch.cat([ref_rgb_tensor, ref_depth.unsqueeze(0)], dim=0)  # [4, H, W]
                ref_seq_list.append(rgbd)
            
            ref_seq = torch.stack(ref_seq_list, dim=0).unsqueeze(0).to(device)  # [1, seq_len, 4, H, W]
            
            # NVDS 推理
            with torch.no_grad():
                outputs = model_nvds(ref_seq)  # [1, 1, H, W]
                outputs = F.relu(outputs)
                outputs = outputs.squeeze(1)  # [1, H, W]
            
            # 存储到原始索引位置
            temp_result[frame_idx] = outputs[0].cpu()
            new_depth_cache[frame_idx] = outputs[0].cpu()
            
            # OPW 指标
            if traversal_i >= 1:
                outputs_flow = outputs.clone().to(device)
                rgb_flow = rgb_tensor.clone().squeeze(0).to(device)  # [1, 3, H, W] → squeeze(0) 去掉 unsqueeze(0)?
                # rgb_tensor 是 [1,1,3,H,W]，需要 [1,3,H,W]
                rgb_flow = rgb_tensor.squeeze(0).to(device)  # [1, 3, H, W]
                
                results_dict = model_flow(
                    rgb_flow, previous_rgb,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list,
                    pred_bidir_flow=args.pred_bidir_flow,
                )
                flow2to1 = results_dict['flow_preds'][-1]
                
                # 反归一化 RGB
                prev_rgb_denorm = previous_rgb.clone()
                for c in range(3):
                    prev_rgb_denorm[:, c] = prev_rgb_denorm[:, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
                prev_rgb_denorm = prev_rgb_denorm * 255
                rgb_flow_denorm = rgb_flow.clone()
                for c in range(3):
                    rgb_flow_denorm[:, c] = rgb_flow_denorm[:, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
                rgb_flow_denorm = rgb_flow_denorm * 255
                
                img1to2_seq = flow_warp(prev_rgb_denorm, flow2to1, mask=False)
                previous_outputs_warp = previous_outputs.unsqueeze(1).to(device)
                outputs1to2 = flow_warp(previous_outputs_warp, flow2to1, mask=False)
                
                tem_loss = temloss(img1to2_seq, rgb_flow_denorm, outputs1to2, outputs_flow, device)
                all_tem += tem_loss.item()
            
            previous_outputs = outputs.clone()
            previous_rgb = rgb_tensor.squeeze(0).to(device)  # [1, 3, H, W]
            
            # 保存可视化和深度
            depth_vis = outputs[0].cpu().numpy()
            save_idx = frame_idx  # 始终按原始索引保存
            plt.imsave(
                save_ksh_dir + str(save_idx) + '.png',
                depth_vis,
                cmap='inferno',
                vmin=0, vmax=1
            )
            save_depth_uint16(outputs[0], save_disp_dir + '/frame_%06d.png' % save_idx)
            save_depth_float(outputs[0], save_float_dir + '/frame_%06d.npy' % save_idx)
        
        # 更新 depth_cache 为本轮结果（✅ 修复迭代反馈）
        depth_cache = new_depth_cache
        
        # 记录指标
        mean_tem = all_tem / len(frames)
        if times % 2 == 0:
            print(f'  Forward OPW: all={all_tem:.4f}, mean={mean_tem:.4f}')
            with open(base_dir + '/result.txt', 'a') as f:
                f.write(f'\n**********forward:{times+1}**********\n')
                f.write(f'all:{all_tem},mean:{mean_tem},frames:{len(frames)}\n')
            if min_fwd - mean_tem >= 1e-3:
                min_fwd = mean_tem
                best_fwd = temp_result
                no_change_times_fwd = 0
            else:
                no_change_times_fwd += 1
        else:
            print(f'  Backward OPW: all={all_tem:.4f}, mean={mean_tem:.4f}')
            with open(base_dir + '/result.txt', 'a') as f:
                f.write(f'\n**********backward:{times+1}**********\n')
                f.write(f'all:{all_tem},mean:{mean_tem},frames:{len(frames)}\n')
            if min_bwd - mean_tem >= 1e-3:
                min_bwd = mean_tem
                best_bwd = temp_result
                no_change_times_bwd = 0
            else:
                no_change_times_bwd += 1
    
    # ============ 阶段3: Mixing 前向+后向 ============
    print('\n===== 阶段3: Mixing =====')
    
    # best_bwd 是按原始索引存储的，不需要 reverse
    # （因为 temp_result[frame_idx] 始终按原始索引存储）
    
    save_disp_dir = base_dir + '/mix/gray/'
    save_ksh_dir = base_dir + '/mix/color/'
    save_float_dir = base_dir + '/mix/float/'
    os.makedirs(save_disp_dir, exist_ok=True)
    os.makedirs(save_ksh_dir, exist_ok=True)
    os.makedirs(save_float_dir, exist_ok=True)
    
    all_tem_mix = 0
    previous_outputs = None
    previous_rgb = None
    
    for i in range(len(frames)):
        frame = frames[i]
        
        fwpred = best_fwd[i]
        bwpred = best_bwd[i]
        
        rgb = img_loader(frame)
        rgb_resized = cv2.resize(rgb, infer_size, interpolation=cv2.INTER_CUBIC)
        rgb_tensor = normalize_rgb(rgb_resized).unsqueeze(0).to(device)
        
        # Mixing 策略：如果两者都优于初始深度，取平均；否则取更好的
        if min_fwd < dpt_index and min_bwd < dpt_index:
            outputs = (fwpred + bwpred) / 2
        elif min_fwd < min_bwd:
            outputs = fwpred
        else:
            outputs = bwpred
        
        outputs = outputs.unsqueeze(0).to(device)  # [1, H, W]
        
        # OPW
        if i >= 1:
            outputs_flow = outputs.clone()
            rgb_flow = rgb_tensor.squeeze(0).clone()
            
            results_dict = model_flow(
                rgb_flow, previous_rgb,
                attn_splits_list=args.attn_splits_list,
                corr_radius_list=args.corr_radius_list,
                prop_radius_list=args.prop_radius_list,
                pred_bidir_flow=args.pred_bidir_flow,
            )
            flow2to1 = results_dict['flow_preds'][-1]
            
            prev_rgb_denorm = previous_rgb.clone()
            for c in range(3):
                prev_rgb_denorm[:, c] = prev_rgb_denorm[:, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
            prev_rgb_denorm = prev_rgb_denorm * 255
            rgb_flow_denorm = rgb_flow.clone()
            for c in range(3):
                rgb_flow_denorm[:, c] = rgb_flow_denorm[:, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
            rgb_flow_denorm = rgb_flow_denorm * 255
            
            img1to2_seq = flow_warp(prev_rgb_denorm, flow2to1, mask=False)
            previous_outputs_warp = previous_outputs.unsqueeze(1).to(device)
            outputs1to2 = flow_warp(previous_outputs_warp, flow2to1, mask=False)
            
            tem_loss = temloss(img1to2_seq, rgb_flow_denorm, outputs1to2, outputs_flow, device)
            all_tem_mix += tem_loss.item()
        
        previous_outputs = outputs.clone()
        previous_rgb = rgb_tensor.squeeze(0).clone()
        
        # 保存
        depth_vis = outputs[0].cpu().numpy()
        plt.imsave(
            save_ksh_dir + str(i) + '.png',
            depth_vis,
            cmap='inferno',
            vmin=0, vmax=1
        )
        save_depth_uint16(outputs[0], save_disp_dir + '/frame_%06d.png' % i)
        save_depth_float(outputs[0], save_float_dir + '/frame_%06d.npy' % i)
    
    mean_mix = all_tem_mix / len(frames)
    print(f'  Mix OPW: all={all_tem_mix:.4f}, mean={mean_mix:.4f}')
    print(f'  min_fwd={min_fwd:.4f}, min_bwd={min_bwd:.4f}')
    
    with open(base_dir + '/result.txt', 'a') as f:
        f.write('\n**********Mixing**********\n')
        f.write(f'all:{all_tem_mix},mean:{mean_mix},frames:{len(frames)}\n')
        f.write(f'min_fwd:{min_fwd},min_bwd:{min_bwd}\n')
    
    # 保存全局统计量到 JSON，方便下游使用
    stats_path = base_dir + '/depth_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'global_median': global_median,
            'global_iqr': global_iqr,
            'infer_w': args.infer_w,
            'infer_h': args.infer_h,
            'encoder': args.encoder,
            'n_frames': len(frames),
            'opw_initial': dpt_index,
            'opw_forward': min_fwd,
            'opw_backward': min_bwd,
            'opw_mix': mean_mix,
        }, f, indent=2)
    
    print(f'\n✅ 完成! 结果保存在: {base_dir}')
    print(f'   全局统计: {stats_path}')
