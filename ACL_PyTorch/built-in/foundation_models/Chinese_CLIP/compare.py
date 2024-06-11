# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
from PIL import Image
from pathlib import Path

import torch
import numpy as np
from torchvision.transforms import Compose, Resize, InterpolationMode
from ais_bench.infer.interface import InferSession

from cn_clip.clip.model import convert_weights, CLIP
import cn_clip.clip as clip
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.clip.utils import image_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Image batch size."
    )    
    parser.add_argument(
        "--resume",
        default="./models/clip_cn_vit-b-16.pt",
        type=str,
        help="path to latest checkpoint (default: none)",
    )    
    parser.add_argument(
        "--npu-device", type=int, default=0, help="Npu device ID."
    ) 
    parser.add_argument(
        "--txt-model-path", 
        type=str, 
        default=None, 
        help="path to img om model."
    )    
    parser.add_argument(
        "--img-model-path", 
        type=str, 
        default=None, 
        help="path to img om model."
    )         
    args = parser.parse_args()

    return args


def _convert_to_rgb(image):
    return torch.tensor(np.array(image.convert('RGB')).astype("uint8"))


def image_transform_wo_normalize(image_size=224):
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
    ])
    return transform


def cpu_infer(args, model_info, txt, img):
    # get model
    model = CLIP(**model_info)
    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    convert_weights(model)
    convert_models_to_fp32(model)  
    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # Map model to be loaded to specified device.
    checkpoint = torch.load(args.resume, map_location='cpu')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    model.eval()

    # cpu infer
    with torch.no_grad():
        assert args.txt_model_path is not None or args.img_model_path is not None, "txt_model_path and img_model_path cannot both be None"
        assert args.txt_model_path is None or args.img_model_path is None, "txt_model_path and img_model_path cannot both passing values"
        if args.txt_model_path:
            out = model(None, txt)
        elif args.img_model_path:
            out = model(img, None) 
    return out


def om_infer(args, txt, img):
    assert args.txt_model_path is not None or args.img_model_path is not None, "txt_model_path and img_model_path cannot both be None"
    assert args.txt_model_path is None or args.img_model_path is None, "txt_model_path and img_model_path cannot both passing values"
    if args.txt_model_path:
        session = InferSession(args.npu_device, args.txt_model_path)
        out = session.infer([txt])
    elif args.img_model_path:
        session = InferSession(args.npu_device, args.img_model_path)
        out = session.infer([img])
    return torch.tensor(out[0])


if __name__ == "__main__":
    args = parse_args()

    # Get model config.
    vision_model_config_file = Path(__file__).parent / \
          f"Chinese-CLIP/cn_clip/clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = Path(__file__).parent / \
          f"Chinese-CLIP/cn_clip/clip/model_configs/{args.text_model.replace('/', '-')}.json"
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        for k, v in json.load(ft).items():
            model_info[k] = v

    # build inputs
    text = clip.tokenize(["皮卡丘"], context_length=512).expand(args.batch_size, 512)
    preprocess = image_transform(model_info["image_resolution"])
    image = preprocess(Image.open("Chinese-CLIP/examples/pokemon.jpeg")).unsqueeze(0).expand(args.batch_size,3,224,224)
    preprocess = image_transform_wo_normalize(model_info["image_resolution"])
    image_wo_normalize = preprocess(Image.open("Chinese-CLIP/examples/pokemon.jpeg")).unsqueeze(0).expand(args.batch_size,224,224,3)

    x = cpu_infer(args, model_info, text, image)
    y = om_infer(args, text, image_wo_normalize)

    similarity = torch.cosine_similarity(torch.tensor(x).reshape(-1), y.reshape(-1), dim=0)
    print(f"模型输出的余弦相似度为：{similarity}")