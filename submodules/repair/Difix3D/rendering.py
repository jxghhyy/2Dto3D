import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from .src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import shutil

def set_difix3d(ref=True):
    if ref:
        pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    else:
        pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe = pipe.to("cuda")
    return pipe

def difix3d_processing(input_image, ref_image, pipe, num_inference_steps=1, timesteps=199, ref=True):
    prompt = "Fix black edges and holes"
    if ref:
        output_image = pipe(prompt, image=input_image, num_inference_steps=num_inference_steps, timesteps=[timesteps], ref_image=ref_image, guidance_scale=0.0).images[0]
    else:
        output_image = pipe(prompt, image=input_image, num_inference_steps=num_inference_steps, timesteps=[timesteps], guidance_scale=0.0).images[0]
    torch.cuda.empty_cache()
    return output_image
