# Copyright 2024 Huawei Technologies Co., Ltd

from diffusers import AutoPipelineForText2Image
import torch
import torch_npu
import os
import datetime

output_path = "./sdxl_turbo_infer"
os.makedirs(output_path, exist_ok=True)

#需要 source /usr/local/Ascend/ascend-toolkit/set_env.sh
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("start time: " + time)

pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline_text2image = pipeline_text2image.to("npu")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
seed_list=[8,23,42,1334]
for i in seed_list:  
                 generator = torch.Generator(device="cpu").manual_seed(i)
                 image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=1,generator=generator,).images[0]
                 image.save(f"{output_path}/turbo_NPU-{i}.png")

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("end time: " + time)

