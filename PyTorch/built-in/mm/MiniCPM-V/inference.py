#!/usr/bin/env python
# encoding: utf-8
from PIL import Image
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
from npu_patch.utils import is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import npu_patch

    torch.npu.config.allow_internal_format = False

# README, How to run demo on different devices

# For Nvidia GPUs.
# python web_demo_2.5.py --device cuda

# For Mac with MPS (Apple silicon or AMD GPUs).
# PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo_2.5.py --device mps

# Argparser
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--device', type=str, default='npu', help='cuda or mps or npu')
parser.add_argument('--model_path', type=str)
parser.add_argument('--prompt', type=str)
parser.add_argument('--image', type=str)
args = parser.parse_args()
device = args.device
assert device in ['cuda', 'mps', 'npu']

# Load model
model_path = args.model_path
if 'int4' in model_path:
    if device == 'mps':
        print('Error: running int4 model with bitsandbytes on Mac is not supported right now.')
        exit()
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
else:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

image = Image.open(args.image).convert('RGB')
question = args.prompt
msgs = [{'role': 'user', 'content': question}]

params = {
    'sampling': False,
    'num_beams': 1,
    'repetition_penalty': 1,
    "max_new_tokens": 896
}

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    **params
)

print(res)
