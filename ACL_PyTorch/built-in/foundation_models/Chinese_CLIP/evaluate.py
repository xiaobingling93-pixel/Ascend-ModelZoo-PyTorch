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


'''
This script refers to https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/eval/zeroshot_evaluation.py
'''

import os
import argparse
from pathlib import Path
import json
import time
from tqdm import tqdm

import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, Resize, InterpolationMode
from torch_npu.contrib import transfer_to_npu
from ais_bench.infer.interface import InferSession

from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.clip import tokenize
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.eval.data import get_zeroshot_dataset, _preprocess_text
from cn_clip.eval.cvinw_zeroshot_templates import (
    openai_templates,
    flower_templates,
    food_templates,
    aircraft_templates,
    eurosat_templates,
    country211_templates,
)


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
        "--label-file",
        type=str,
        help="file for labels",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to the test set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="Specified dataset.",
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )    
    parser.add_argument(
        "--context-length", 
        type=int, 
        default=52,
        help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--resume",
        default="./models/clip_cn_vit-b-16.pt",
        type=str,
        help="path to latest checkpoint (default: none)",
    )    
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for ImageNet dataloader."
    )
    parser.add_argument(
        "--npu-device", type=int, default=0, help="Npu device ID."
    ) 
    parser.add_argument(
        "--save-class-embeddings", 
        action="store_true",
        help="Compute and save class embeddings."
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
    return np.array(image.convert('RGB')).astype("uint8")


def image_transform(image_size=224):
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
    ])
    return transform


def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [_preprocess_text(template(classname)) for template in templates]  # format with class
            texts = tokenize(texts, context_length=args.context_length).to(args.npu_device)  # tokenize
            class_embeddings = model(None, texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.npu_device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(session, classifier, dataloader, args):
    total_logits = []
    total_targets = []
    infer_time = []
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader):

            total_targets.append(target)

            # om model infer
            start_time = time.time()
            bs = images.shape[0]
            if bs < args.img_batch_size:
                p1d = (0, 0, 0, 0, 0, 0, 0, args.img_batch_size - images.shape[0])
                images = F.pad(images, p1d, "constant", 0)
            image_features = torch.tensor(session.infer([images.numpy().astype("uint8")])[0])
            if bs < args.img_batch_size:
                image_features = image_features[:bs, :]
            infer_time.append(time.time() - start_time)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ classifier).softmax(dim=-1)
            total_logits.append(logits)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 1))
            top1 += acc1
            n += images.size(0)

    outputs = torch.cat(total_logits, dim=0)
    top1 = top1 / n

    return top1, outputs


if __name__ == "__main__":
    args = parse_args()

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")


    # Get model config.
    vision_model_config_file = Path(__file__).parent.parent / \
          f"Chinese-CLIP/cn_clip/clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = Path(__file__).parent.parent / \
          f"Chinese-CLIP/cn_clip/clip/model_configs/{args.text_model.replace('/', '-')}.json"
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        for k, v in json.load(ft).items():
            model_info[k] = v

    if args.save_class_embeddings:
        model = CLIP(**model_info)

        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        convert_weights(model)
        convert_models_to_fp32(model)  
        model.cuda(args.npu_device)
        
        # Resume from a checkpoint.
        print("Begin to load model checkpoint from {}.".format(args.resume))
        assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        model.load_state_dict(sd)
        print(
            f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)"
        ) 
        model.eval()

        with open(args.label_file, 'r', encoding="utf8") as f:
            classnames = [line.strip() for line in f.readlines()]
        template_dict = {
            "fgvc-aircraft-2013b-variants102": aircraft_templates,
            "food-101": food_templates,
            "oxford-flower-102": flower_templates,
            "eurosat_clip": eurosat_templates,
            "resisc45_clip": eurosat_templates,
            "country211": country211_templates,
            "openai": openai_templates,
        }
        if args.dataset in template_dict.keys():
            templates = template_dict[args.dataset]
        else:
            templates = template_dict['openai']

        classifier = zero_shot_classifier(model, classnames, templates, args)
        torch.save(classifier.cpu(), "models/classifier.pt")
        print('Compute and save ensembled class embeddings')
    
    else:
        start_time = time.time()
        img_session = InferSession(args.npu_device, args.img_model_path)
        
        # Get eval data.
        data = {}
        data[args.dataset] = get_zeroshot_dataset(
            args, image_transform(model_info["image_resolution"])
        )

        # Inference and evaluation
        classifier = torch.load("models/classifier.pt")
        top1, logits = run(img_session, classifier, data[args.dataset].dataloader, args)
        print("Total evaluation time:{}s".format(time.time() - start_time))

        # Result
        results = {}
        results["zeroshot-top1"] = top1
        print('Result:')
        print(", ".join(["{}: {}".format(k, v) for k, v in results.items()]))
        print('Finished.')