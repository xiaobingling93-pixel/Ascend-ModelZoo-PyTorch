# Copyright 2024 Huawei Technologies Co., Ltd

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
import random
import os
import numpy as np
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_path = "/diffusion/model"

#需要  source /usr/local/Ascend/ascend-toolkit/set_env.sh

output_path = "./sdxl_controlnet_NPU"
os.makedirs(output_path, exist_ok=True)
def seed_all(seed=1234, mode=False, is_gpu=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    if is_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    else:
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)

seed_all()


controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float32
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# memory optimization.
pipe.enable_model_cpu_offload()
control_image = load_image("./conditioning_image_1.png")

prompts = dict()
prompts["masterpiece, best quality, Cute dragon creature, pokemon style, night, moonlight, dim lighting"] = "deformed, disfigured, underexposed, overexposed, rugged, (low quality), (normal quality),"
prompts["masterpiece, best quality, Pikachu walking in beijing city, pokemon style, night, moonlight, dim lighting"] = "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),"
prompts["masterpiece, best quality, red panda , pokemon style, evening light, sunset, rim lighting"] = "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),"
prompts["masterpiece, best quality, Photo of (Lion:1.2) on a couch, flower in vase, dof, film grain, crystal clear, pokemon style, dark studio"] = "deformed, disfigured, underexposed, overexposed, "
prompts["masterpiece, best quality, siberian cat pokemon on river, pokemon style, evening light, sunset, rim lighting, depth of field"] = "deformed, disfigured, underexposed, overexposed, "
prompts["masterpiece, best quality, pig, Exquisite City, (sky:1.3), (Miniature tree:1.3), Miniature object, many flowers, glowing mushrooms, (creek:1.3), lots of fruits, cute colorful animal protagonist, Firefly, meteor, Colorful cloud, pokemon style, Complicated background, rainbow,"] = "Void background,black background,deformed, disfigured, underexposed, overexposed, "
prompts["masterpiece, best quality, (pokemon), a cute pikachu, girl with glasses, (masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2),"] = "(low quality), (normal quality), (monochrome), lowres, extra fingers, fewer fingers, (watermark), "
prompts["masterpiece, best quality, sugimori ken \(style\), (pokemon \(creature\)), pokemon electric type, grey and yellow skin, mechanical arms, cyberpunk city background, night, neon light"] = "(worst quality, low quality:1.4), watermark, signature, deformed, disfigured, underexposed, overexposed, "
#设置随机数种子
seed_list=[8,23,42,1334]

#输出图片
for prompt_key, negative_prompt_key in prompts.items():
    for i in seed_list:    
        generator = torch.Generator(device="cpu").manual_seed(i)
        image = pipe(prompt=prompt_key, negative_prompt=negative_prompt_key, generator=generator,image=control_image,num_inference_steps=50).images
        image[0].save(f"{output_path}/{prompt_key[26:40]}-{i}.png")

