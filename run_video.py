import argparse
import cv2
import torch.nn.functional as F
import glob
import numpy as np
import os
import time
import torch
from torchvision.transforms import Compose

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='enable fp16 inference on cuda')
    parser.add_argument('--warmup-iters', type=int, default=10, help='number of warmup iterations on cuda')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    if args.fp16 and DEVICE == 'cuda':
        depth_anything = depth_anything.half()

    transform = Compose([
        Resize(
            width=args.input_size,
            height=args.input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if DEVICE == 'cuda' and args.warmup_iters > 0:
        warmup_dtype = torch.float16 if args.fp16 else torch.float32
        dummy_input = torch.randn(1, 3, args.input_size, args.input_size, device=DEVICE, dtype=warmup_dtype)
        with torch.inference_mode():
            for _ in range(args.warmup_iters):
                _ = depth_anything(dummy_input)
        torch.cuda.synchronize()
    
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 50
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        
        if args.pred_only: 
            output_width = frame_width
        else: 
            output_width = frame_width * 2 + margin_width
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        stage_times = {
            'read_frame': 0.0,
            'infer_depth': 0.0,
            'postprocess_depth': 0.0,
            'compose_frame': 0.0,
            'write_frame': 0.0,
        }
        processed_frames = 0
        video_start_time = time.perf_counter()
        
        while raw_video.isOpened():
            t0 = time.perf_counter()
            ret, raw_frame = raw_video.read()
            stage_times['read_frame'] += time.perf_counter() - t0
            if not ret:
                break

            processed_frames += 1

            t0 = time.perf_counter()
            h, w = raw_frame.shape[:2]
            image = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
            if args.fp16 and DEVICE == 'cuda':
                image = image.half()

            with torch.inference_mode():
                if DEVICE == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.fp16):
                        depth = depth_anything(image)
                else:
                    depth = depth_anything(image)

            depth = F.interpolate(depth[:, None], (h, w), mode='bilinear', align_corners=True)[0, 0]
            depth = depth.detach().float().cpu().numpy()
            stage_times['infer_depth'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            stage_times['postprocess_depth'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if args.pred_only:
                frame_to_write = depth
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                frame_to_write = cv2.hconcat([raw_frame, split_region, depth])
            stage_times['compose_frame'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            out.write(frame_to_write)
            stage_times['write_frame'] += time.perf_counter() - t0
        
        raw_video.release()
        out.release()

        total_elapsed = time.perf_counter() - video_start_time
        avg_fps = processed_frames / total_elapsed if total_elapsed > 0 else 0.0
        stage_total = sum(stage_times.values())

        print(f'[{os.path.basename(filename)}] processed_frames={processed_frames}, total_time={total_elapsed:.3f}s')
        for stage_name, stage_time in stage_times.items():
            stage_ratio = (stage_time / stage_total * 100.0) if stage_total > 0 else 0.0
            print(f'  {stage_name}: {stage_time:.3f}s ({stage_ratio:.1f}%)')
        print(f'  avg_fps: {avg_fps:.2f} frames/s')
