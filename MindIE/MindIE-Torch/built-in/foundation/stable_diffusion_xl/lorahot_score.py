# Copyright 2023 Huawei Technologies Co., Ltd
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
import json
import time
import argparse
import logging

import open_clip
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

printlog = logging.getLogger()
printlog.addHandler(logging.StreamHandler())
printlog.setLevel(logging.INFO)


# single image
def cos_similarity(model_clip, preprocess, image_file1, image_file2, device):
    img1 = preprocess(Image.open(image_file1)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(image_file2)).unsqueeze(0).to(device)

    img_ft1 = model_clip.encode_image(img1).float()
    img_ft2 = model_clip.encode_image(img2).float()

    score = F.cosine_similarity(img_ft1, img_ft2).squeeze()

    return score.cpu()


def main():
    args = parse_arguments()
    
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)
    
    t_b = time.time()
    printlog.info("Load clip model...") 
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.model_weights_path, device=device)
    model_clip.eval()
    printlog.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")

    with os.fdopen(os.open(args.image_info_wo_lorahot, os.O_RDONLY), "r") as f:
        image_info_wo_lorahot = json.load(f)
    
    with os.fdopen(os.open(args.image_info_lorahot, os.O_RDONLY), "r") as f:
        image_info_lorahot = json.load(f)

    t_b = time.time()
    printlog.info("Calc cos similarity score...") 
    all_scores = []
    info_length = len(image_info_wo_lorahot)
    for i in range(info_length):
        
        image_file1 = image_info_wo_lorahot[i]['images']
        image_file2 = image_info_lorahot[i]['images']
        prompt = image_info_wo_lorahot[i]['prompt']
        printlog.info(f"[{i + 1}/{len(image_info_wo_lorahot)}] {prompt}")

        image_scores = cos_similarity(model_clip, 
                                  preprocess,
                                  image_file1, 
                                  image_file2, 
                                  device)

        printlog.info(f"cos similarity scores: {image_scores}")

        all_scores.append(image_scores)
    printlog.info(f">done. elapsed time: {(time.time() - t_b):.3f} s")

    average_score = np.average(all_scores)
    printlog.info("====================================")
    printlog.info(f"average score: {average_score:.3f}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="device for torch.",
    )
    parser.add_argument(
        "--image_info_wo_lorahot",
        type=str,
        default="./image_info_wo_lorahot.json",
        help="Image_info_wo_lorahot.json file.",
    )
    parser.add_argument(
        "--image_info_lorahot",
        type=str,
        default="./image_info_lorahot.json",
        help="Image_info_lorahot.json file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14",
        help="open clip model name",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
        help="open clip model weights",
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()