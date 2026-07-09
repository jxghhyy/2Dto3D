import os
import sys
import torch
from thop import profile, clever_format

def align_to_14(size):
    """将尺寸对齐到最近的14的倍数"""
    return ((size + 13) // 14) * 14

def calculate_depth_anything_flops():
    print("=" * 60)
    print(" Depth-Anything-V2 算力分析工具")
    print(" 注意: 输入尺寸必须是14的倍数 (ViT patch size=14)")
    print("=" * 60)
    
    try:
        from submodules.depth.dav2.depth_anything_v2.dpt import DepthAnythingV2
        print(" Depth-Anything-V2 导入成功")
    except ImportError:
        print(" 未找到 Depth-Anything-V2 包")
        print("   用ResNet50作为参考模型")
        
        from torchvision.models import resnet50
        model = resnet50()
        model_name = "ResNet50 (参考模型)"
        is_ref = True
    else:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        
        model = DepthAnythingV2(**model_configs['vits'])
        model_name = "Depth-Anything-V2 Small"
        is_ref = False
        
        weight_path = 'checkpoints/depth_anything_v2_vits.pth'
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print(f" 权重加载成功: {weight_path}")
        else:
            print(" 未找到权重，使用随机初始化（FLOPs计算不受影响）")
    
    model.eval()
    print()
    
    resolutions_raw = [
        (518, 294, "标准测试尺寸"),
        (640, 480, "VGA尺寸"),
        (1280, 720, "720p高清"),
        (1920, 1080, "1080p全高清"),
    ]
    
    resolutions = []
    for w, h, desc in resolutions_raw:
        w_aligned = align_to_14(w)
        h_aligned = align_to_14(h)
        
        if w != w_aligned or h != h_aligned:
            desc += f" 对齐到 {w_aligned}{h_aligned}"
        
        resolutions.append((w_aligned, h_aligned, desc))
    
    print(" 测试分辨率（均已对齐到14倍数）:")
    for w, h, desc in resolutions:
        print(f"   {w}{h} ({desc})")
    print()
    
    results = []
    
    for w, h, desc in resolutions:
        input_tensor = torch.randn(1, 3, h, w)
        
        macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
        flops = macs * 2
        
        flops_str, params_str, macs_str = clever_format([flops, params, macs], "%.3f")
        
        results.append({
            'resolution': f"{w}{h}",
            'w': w,
            'h': h,
            'desc': desc,
            'flops': flops,
            'flops_str': flops_str,
            'macs': macs,
            'macs_str': macs_str,
            'params': params,
            'params_str': params_str,
        })
        
        print(f" {w}{h}: {flops_str} FLOPs")
    
    print("\n" + "=" * 60)
    print(f" {model_name} - 详细算力报告")
    print("=" * 60)
    print(f"\n 模型参数量: {results[0]['params_str']}")
    print(f" ViT Patch Size: 1414")
    print()
    
    print("-" * 60)
    print(f"{'分辨率':<15} {'FP32 FLOPs':<15} {'30FPS(INT8)':<15} {'50FPS(INT8)':<15}")
    print("-" * 60)
    
    for r in results:
        tops_30fps = (r['flops'] * 30 / 1e12) * 0.5
        tops_50fps = (r['flops'] * 50 / 1e12) * 0.5
        
        print(f"{r['resolution']:<15} "
              f"{r['flops']/1e9:>6.2f} GFLOPs   "
              f"{tops_30fps:>6.3f} TOPS  "
              f"{tops_50fps:>6.3f} TOPS")
    
    print("-" * 60)
    
    print("\n" + "=" * 60)
    print(" 1080p 详细分析")
    print("=" * 60)
    
    r_1080p = next(r for r in results if '1080' in r['desc'])
    
    print(f"\n原始: 19201080  对齐后: {r_1080p['resolution']}")
    print(f"高度方向: 1080  {r_1080p['h']} (增加了 {r_1080p['h']-1080} 像素)")
    print(f"宽度方向: 1920  {r_1080p['w']} (1920是14的倍数: {1920%14 == 0})")
    
    print(f"\n单帧计算量: {r_1080p['flops']/1e9:.2f} GFLOPs (FP32)")
    required = (r_1080p['flops'] * 50 / 1e12) * 0.5
    print(f"50FPS需求: {required:.3f} TOPS (INT8)")
    
    print("\n" + "=" * 60)
    print(" 与 '4T算力' 需求对比")
    print("=" * 60)
    
    budget_tops = 4.0
    required_tops = (r_1080p['flops'] * 50 / 1e12) * 0.5
    
    print(f"\n目标: {r_1080p['resolution']} @ 50 FPS")
    print(f"需要算力: {required_tops:.3f} TOPS (INT8)")
    print(f"预算算力: {budget_tops:.1f} TOPS")
    print(f"算力余量: {budget_tops - required_tops:.3f} TOPS")
    print(f"理论利用率: {required_tops / budget_tops * 100:.1f}%")
    
    real_utilization_rate = 0.5
    real_required = required_tops / real_utilization_rate
    print(f"\n 实际部署时考虑硬件利用率 (50%):")
    print(f"   实际需要: {real_required:.3f} TOPS")
    
    if real_required <= budget_tops:
        print("\n 实际算力充足！")
    else:
        print(f"\n 实际算力可能紧张，需要: {real_required:.3f} TOPS")
        print("   建议进一步优化模型或降低分辨率")
    
    print("\n" + "=" * 60)
    print(" 推荐板卡参考")
    print("=" * 60)
    
    boards = [
        ("RK3588", 6, "NPU，需要算子适配"),
        ("Jetson Xavier NX", 21, " 推荐，性价比最高"),
        ("Jetson Orin NX 8GB", 70, " 稳妥，余量充足"),
        ("Jetson Orin NX 16GB", 100, " 高性能，未来升级无忧"),
    ]
    
    for name, tops, comment in boards:
        status = "" if tops >= required_tops * 3 else ""
        print(f"{status} {name:<25} 标称: {tops:>3} TOPS  {comment}")
    
    print("\n" + "=" * 60)
    print(" 分析完成！")
    print("=" * 60)
    
    if is_ref:
        print("\n 注意: 使用ResNet50作为参考")
        print("   在Depth-Anything-V2目录下运行可获得精确数据")
    
    return results

if __name__ == "__main__":
    calculate_depth_anything_flops()