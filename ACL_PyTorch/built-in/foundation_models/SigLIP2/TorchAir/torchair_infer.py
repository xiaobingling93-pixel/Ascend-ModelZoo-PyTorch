# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
from time import time
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer, pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SigLIP2 performance tests")
    parser.add_argument("--pytorch_ckpt_path", type=str, required=True, help="Path to SigLIP2 model")
    parser.add_argument("--image_path", type=str, default="zero_shot_test_image.jpg", help="Path to input image")
    parser.add_argument("--candidate_labels", type=str, default="2 cats, a plane, a remote", help="Candidate labels for classification")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--loop", type=int, default=10, help="Number of inference loops for performance test")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device ID to use")
    args = parser.parse_args()

    args.candidate_labels = [label.strip() for label in args.candidate_labels.split(",")]

    device = torch.device(f'npu:{args.device_id}')
    torch_npu.npu.set_device(args.device_id)
    
    model = AutoModel.from_pretrained(args.pytorch_ckpt_path).to(device).eval()
    model.optimize_qkv_for_inference()
    image_processor = AutoImageProcessor.from_pretrained(args.pytorch_ckpt_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pytorch_ckpt_path)
    
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backbend = tng.get_npu_backend(compiler_config=config)
    model.vision_model = torch.compile(model.vision_model, dynamic=True, fullgraph=True, backend=npu_backbend)
    model.text_model = torch.compile(model.text_model, dynamic=True, fullgraph=True, backend=npu_backbend)

    image = Image.open(args.image_path).convert("RGB")
    image_inputs = image_processor(images=image, return_tensors="pt").to(device)
    image_inputs = {k: v.repeat(args.batch_size, *([1] * (v.ndim - 1))) for k, v in image_inputs.items()}

    text_inputs = tokenizer(args.candidate_labels, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
    
    image_classifier = pipeline(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        task="zero-shot-image-classification",
        candidate_labels=args.candidate_labels,
        device=device
    )

    with torch.no_grad():
        print("Start compiling...")
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        outputs = image_classifier(image)
        print(outputs)
        print("Warm up done.")

    print("Start performance test...")

    text_times = []
    for _ in range(args.loop):
        torch.npu.synchronize()
        st = time()
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        torch.npu.synchronize()
        text_times.append(time() - st)
    text_perf = round(1 / (sum(text_times) / args.loop), 2)
    print(f"Text inputs shape(labelnums x seqlen): {text_inputs['input_ids'].shape}")
    print(f"Text encoder performance: {text_perf} text/s")

    image_times = []
    for _ in range(args.loop):
        torch.npu.synchronize()
        st = time()
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
        torch.npu.synchronize()
        image_times.append(time() - st)
    image_perf = round(args.batch_size / (sum(image_times) / args.loop), 2)
    print(f"Image inputs shape(bs x channels x height x width): {image_inputs['pixel_values'].shape}")
    print(f"Image encoder performance: {image_perf} image/s")