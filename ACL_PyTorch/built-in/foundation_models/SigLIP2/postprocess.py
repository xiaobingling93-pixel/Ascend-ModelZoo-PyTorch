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

import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def load_ground_truth(gt_path):
    gt = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_key = parts[0].split('.')[0]
            try:
                gt[img_key] = int(parts[1])
            except ValueError:
                continue
    return gt


def load_feature(txt_path):
    try:
        with open(txt_path, 'r') as f:
            line = f.read().strip()
            if not line:
                return None
            values = line.split()
            return np.array([float(x) for x in values], dtype=np.float32)
    except Exception as e:
        print(f"load error {txt_path}: {e}")
        return None


def get_logit_scale_and_bias(model_path):
    model = AutoModel.from_pretrained(model_path).eval()
    logit_scale = model.logit_scale.item()
    logit_bias = model.logit_bias.item()
    return logit_scale, logit_bias


def evaluate(text_feature_result, image_feature_result, gt_file, label_nums, model_path, topk=5):
    try:
        start = time.time()
        
        gt_dict = load_ground_truth(gt_file)
        
        logit_scale, logit_bias = get_logit_scale_and_bias(model_path)
        
        image_feature_files = [f for f in os.listdir(image_feature_result) if f.endswith('.txt')]

        text_feature_file = [f for f in os.listdir(text_feature_result) if f.endswith('.txt')]
        if not text_feature_file:
            raise FileNotFoundError("not found text_feature_file")
        text_feature_path = os.path.join(text_feature_result, text_feature_file[0])
        text_feature = load_feature(text_feature_path)
        hidden_size = text_feature.size // label_nums
        text_feature = text_feature.reshape(label_nums, hidden_size)
        text_feature = text_feature / np.linalg.norm(text_feature, axis=1, keepdims=True)
        
        correct = np.zeros(topk)
        total = 0
        
        for fname in tqdm(image_feature_files, desc="evaluate"):
            base = fname.split('.')[0]
            img_key = base[:base.rfind('_')] if '_' in base else base
            
            if img_key not in gt_dict:
                continue
            
            feat_path = os.path.join(image_feature_result, fname)
            image_feature = load_feature(feat_path)
            image_feature = image_feature / np.linalg.norm(image_feature)
            
            logits = np.dot(image_feature, text_feature.T)
            logits = logits * np.exp(logit_scale) + logit_bias
            logits = 1 / (1 + np.exp(-logits))
            
            top_pred = np.argsort(-logits)[:topk]
            true_label = gt_dict[img_key]
            
            for i in range(topk):
                if top_pred[i] == true_label:
                    correct[i] += 1
                    break
            
            total += 1
        
        accuracy = np.cumsum(correct) / total if total > 0 else np.zeros(topk)
        
        result = {
            "title": "siglip2 zero-shot-image-classification task accuracy",
            "value": [
                {"key": "image nums", "value": str(total)},
                {"key": "label nums", "value": "1000"},
            ]
        }
        for i in range(topk):
            result["value"].append({
                "key": f"Top-{i+1} accuracy",
                "value": f"{accuracy[i] * 100:.2f}%"
            })
        
        print(f"\nimage nums: {total}")
        for i in range(topk):
            print(f"Top-{i+1} accuracy: {accuracy[i] * 100:.2f}%")
        
        return result
    except Exception as e:
        print(f"evaluate error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="postprocessing script")
    parser.add_argument(
        "--text_feature_result", 
        type=str, 
        required=True, 
        help="Directory where text features are stored."
    )
    
    parser.add_argument(
        "--image_feature_result", 
        type=str, 
        required=True, 
        help="Directory where image features are stored."
    )
    
    parser.add_argument(
        "--gt_file", 
        type=str, 
        required=True, 
        help="Path to the ground truth label file."
    )

    parser.add_argument(
        "--label_nums", 
        type=int, 
        required=True, 
        help="Num of dateset's label."
    )

    parser.add_argument(
        "--pytorch_ckpt_path", 
        type=str, 
        required=True, 
        help="Path to the PyTorch model checkpoint."
    )

    args = parser.parse_args()

    result = evaluate(
        text_feature_result=args.text_feature_result,
        image_feature_result=args.image_feature_result,
        gt_file=args.gt_file,
        label_nums=args.label_nums,
        model_path=args.pytorch_ckpt_path
    )

    if result is None:
        print("[ERROR] Evaluation failed, please check the input paths and files.")
    else:
        print("[INFO] Evaluation end.")