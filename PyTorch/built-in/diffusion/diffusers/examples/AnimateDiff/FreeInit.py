# Copyright 2024 Huawei Technologies Co., Ltd

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif
import time
#torch.npu.set_compile_mode(jit_compile=True) 
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16).to("npu")
pipe.scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    clip_sample=False,
    timestep_spacing="linspace",
    steps_offset=1
)

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# enable FreeInit
# Refer to the enable_free_init documentation for a full list of configurable parameters
pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

# run inference
a = time.time()
output = pipe(
    prompt="a panda playing a guitar, on a boat, in the ocean, high quality",
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(666),
)
print("inference time :",time.time()-a)
# disable FreeInit
pipe.disable_free_init()

frames = output.frames[0]
export_to_gif(frames, "animation4.gif")