# Copyright 2024 Huawei Technologies Co., Ltd

import torch
import torch_npu
import time
import torchair as tng
from PIL import Image
from diffusers import UniDiffuserPipeline
from diffusers.utils.import_utils import is_torch_npu_available

def open_graph_mode(pipe):
    npu_backend = tng.get_npu_backend()
    pipe.unet = torch.compile(pipe.unet, backend=npu_backend, dynamic=False)
    pipe.text_encoder = torch.compile(pipe.text_encoder, backend=npu_backend, dynamic=False)
    pipe.text_decoder = torch.compile(pipe.text_decoder, backend=npu_backend, dynamic=False)
    pipe.image_encoder = torch.compile(pipe.image_encoder, backend=npu_backend, dynamic=False)
    pipe.vae = torch.compile(pipe.vae, backend=npu_backend, dynamic=False)
    pipe.clip_tokenizer = torch.compile(pipe.clip_tokenizer, backend=npu_backend, dynamic=False)

device = "npu"
model_id_or_path = "./unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
if is_torch_npu_available():
    pipe.unet.enable_npu_flash_attention()
    open_graph_mode(pipe)
pipe.to(device)

prompt = "an elephant under the sea"
print("=========start warm up============")
for _ in range(10):
    _ = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
print("=========end warm up============")

# Text to image generation
repeat = 20
totaltime = 0
for i in range(repeat):
    starttime = time.time()
    sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
    torch.npu.synchronize()
    totaltime += time.time() - starttime
    t2i_image = sample.images[0]
    t2i_image.save("elephant.png")

print("text to image generation avg time:", totaltime / repeat)