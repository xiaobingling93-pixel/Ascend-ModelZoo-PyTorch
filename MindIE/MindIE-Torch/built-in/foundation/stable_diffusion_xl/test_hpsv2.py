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


import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import requests
from clint.textui import progress
from typing import Union
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map
import hpsv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

model_dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_model(pretrained_path: str):
    if not model_dict:
        model, _, preprocess_val = create_model_and_transforms("ViT-H-14", pretrained=pretrained_path,
                                                               precision='amp',
                                                               device=device,
                                                               jit=False,
                                                               force_quick_gelu=False,
                                                               force_custom_text=False,
                                                               force_patch_dropout=False,
                                                               force_image_size=None,
                                                               pretrained_image=False,
                                                               image_mean=None,
                                                               image_std=None,
                                                               light_augmentation=True,
                                                               aug_cfg={},
                                                               output_dict=True,
                                                               with_score_predictor=False,
                                                               with_region_predictor=False)
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val


def score(img_path: str, pretrained_path: str, prompt_class: str, cp: str = None, hps_version: str = "v2.0") -> list:
    print("prompt_class is:", prompt_class)
    initialize_model(pretrained_path)
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if cp is None:
        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])

    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    img_paths = [os.path.join(img_path, filename) for filename in os.listdir(img_path)]
    all_prompts = hpsv2.benchmark_prompts('all')[prompt_class]

    result = []
    for i, one_img_path in enumerate(img_paths):

        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            if isinstance(one_img_path, str):
                image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
            elif isinstance(one_img_path, Image.Image):
                image = preprocess_val(one_img_path).unsqueeze(0).to(device=device, non_blocking=True)
            else:
                raise TypeError('The type of parameter img_path is illegal.')
            # Process the prompt
            text = tokenizer([all_prompts[i]]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                print("index is", i, ", hps_score is", hps_score)

        result.append(hps_score[0])
    return result


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt_class', type=str, required=True, help='Text prompt')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(root_path, 'HPS_v2_compressed.pt'),
                        help='Path to the model checkpoint')
    parser.add_argument('--pretrained_path', type=str, required=True, help='pretrained path')

    args = parser.parse_args()

    all_hps_score = []
    if args.prompt_class == "all":
        for i in ["anime", "concept-art", "paintings", "photo"]:
            hps_score = score(os.path.join(args.image_path, i), args.pretrained_path, i, args.checkpoint)
            all_hps_score.append(hps_score)
        print('sum HPSv2 score:', all_hps_score)
        print('avg HPSv2 score:', np.sum(all_hps_score) / len(all_hps_score) * 800)
    else:
        hps_score = score(args.image_path, args.pretrained_path, args.prompt_class, args.checkpoint)
        print('sum HPSv2 score:', hps_score)
        print('avg HPSv2 score:', sum(hps_score) / len(hps_score))

