#!/usr/bin/env python
"""
测试 SD 模型能否正常加载和运行
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

# 添加 Mono2Stereo 路径
mono2stereo_path = '/mnt/A/jiangxg/work/2Dto3D/submodules/Mono2Stereo'
if mono2stereo_path not in sys.path:
    sys.path.insert(0, mono2stereo_path)

from marigold.marigold_pipeline import MarigoldPipeline
from stereo.stereo_pipeline import Stereo
from depth.depth_anything_v2.dpt import DepthAnythingV2
from depth.depth_anything_v2.util.transform import MyResize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 测试加载 SD 模型
print("\n" + "="*60)
print("1. 测试加载 MarigoldPipeline")
print("="*60)
try:
    _pipeline_kwargs = {'scale_invariant': True, 'shift_invariant': True}

    # 尝试本地路径
    local_sd_path = os.path.join(mono2stereo_path, "checkpoint/stable-diffusion-2")
    print(f"尝试加载: {local_sd_path}")
    painter = MarigoldPipeline.from_pretrained(local_sd_path, **_pipeline_kwargs).to(device)
    print("✅ MarigoldPipeline 加载成功!")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("\n尝试从 HuggingFace 下载...")
    try:
        painter = MarigoldPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", **_pipeline_kwargs
        ).to(device)
        print("✅ 从 HuggingFace 加载成功!")
    except Exception as e2:
        print(f"❌ HuggingFace 也失败: {e2}")
        sys.exit(1)

# 2. 测试加载深度模型
print("\n" + "="*60)
print("2. 测试加载 Depth-Anything-V2")
print("="*60)
try:
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    depth_model = DepthAnythingV2(**model_configs['vits'])

    # 尝试多个路径
    ckpt_paths = [
        "/mnt/A/jiangxg/work/2Dto3D/submodules/Mono2Stereo/depth/checkpoints/depth_anything_v2_vits.pth",
        "/mnt/A/jiangxg/work/2Dto3D/checkpoints/depth_anything_v2_vits.pth",
    ]

    ckpt_loaded = False
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            print(f"加载权重: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            depth_model.load_state_dict(state_dict)
            ckpt_loaded = True
            break

    if not ckpt_loaded:
        print("⚠️  找不到本地权重，尝试自动下载...")
        from torch.hub import load_state_dict_from_url
        url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
        state_dict = load_state_dict_from_url(url, map_location=device)
        depth_model.load_state_dict(state_dict)

    depth_model = depth_model.to(device).eval()
    print("✅ Depth-Anything-V2 加载成功!")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试创建 Stereo 模型
print("\n" + "="*60)
print("3. 测试创建 Stereo 模型")
print("="*60)
try:
    from torch.nn import Conv2d, Parameter

    # 修改 UNet 输入通道为 12
    if 12 != painter.unet.config["in_channels"]:
        print("修改 UNet 输入通道为 12...")
        _weight = painter.unet.conv_in.weight.clone()
        _bias = painter.unet.conv_in.bias.clone()
        _weight = _weight.repeat((1, 3, 1, 1))
        _weight *= 0.33333
        _new_conv_in = Conv2d(12, painter.unet.conv_in.out_channels,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        painter.unet.conv_in = _new_conv_in
        painter.unet.config["in_channels"] = 12

    # 创建一个空的 args
    class Args:
        pass
    args = Args()

    stereo_model = Stereo(depth_model=depth_model, painter=painter, args=args).to(device)
    print("✅ Stereo 模型创建成功!")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试加载 mono2stereo 权重
print("\n" + "="*60)
print("4. 测试加载 mono2stereo 权重")
print("="*60)
try:
    ckpt_path = "/mnt/A/jiangxg/work/2Dto3D/submodules/Mono2Stereo/checkpoint/mono2stereo.ckpt"
    if os.path.exists(ckpt_path):
        print(f"加载权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'model' in ckpt:
            stereo_model.load_state_dict(ckpt['model'], strict=False)
            print("✅ 从 'model' key 加载成功")
        else:
            stereo_model.load_state_dict(ckpt, strict=False)
            print("✅ 直接加载成功")
    else:
        print(f"⚠️  找不到权重: {ckpt_path}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

stereo_model.eval()

# 5. 测试单张图片推理
print("\n" + "="*60)
print("5. 测试单张图片推理")
print("="*60)
try:
    # 找一张测试图
    test_img = None
    for root, dirs, files in os.walk("/mnt/A/jiangxg/dataset/mono2stereo-test"):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_img = os.path.join(root, f)
                break
        if test_img:
            break

    if test_img is None:
        print("⚠️  找不到测试图片，跳过")
    else:
        print(f"测试图片: {test_img}")

        # 预处理
        img_bgr = cv2.imread(test_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        transform = Compose([
            MyResize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # 为 SD 准备
        new_size = (1280, 800)
        left_image_pil = Image.fromarray(img_rgb).convert("RGB").resize(new_size)
        rgb_left_norm = pil_to_tensor(left_image_pil).unsqueeze(0)
        rgb_left_norm: torch.Tensor = (rgb_left_norm / 255.0 * 2.0 - 1.0).to(device)

        # 为深度准备
        rgb_for_depth = transform({'image': img_rgb.astype(int)/255.0})['image']
        rgb_for_depth = torch.from_numpy(rgb_for_depth).unsqueeze(0).to(device)

        print("开始推理...")
        with torch.no_grad():
            viewpoint_latent = stereo_model.generate_viewpoint(rgb_left_norm, rgb_for_depth)
            print(f"✅ generate_viewpoint 成功，latent shape: {viewpoint_latent.shape}")

            seed = 42
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            pipe_out = stereo_model.painter(
                left_image_pil,
                viewpoint_latent,
                denoising_steps=5,
                ensemble_size=1,
                processing_res=768,
                match_input_res=True,
                generator=generator,
                batch_size=1,
                color_map=None,
                show_progress_bar=False,
                resample_method='bilinear',
            )

            pred_right = pipe_out.right_visual
            print(f"✅ painter 成功，输出 shape: {pred_right.shape}")

        # 保存测试结果
        output_path = "/mnt/A/jiangxg/work/2Dto3D/test_sd_output.png"
        cv2.imwrite(output_path, cv2.cvtColor(pred_right, cv2.COLOR_RGB2BGR))
        print(f"✅ 结果已保存: {output_path}")

except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 测试完成!")
print("="*60)
