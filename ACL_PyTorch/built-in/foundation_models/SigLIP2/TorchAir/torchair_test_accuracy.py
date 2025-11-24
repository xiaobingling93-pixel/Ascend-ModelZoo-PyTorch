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
import argparse
import time
import ast
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def load_imagenet_classnames(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        d = ast.literal_eval(f.read())
    if len(d) != 1000:
        raise ValueError(f"Expected 1000 classes, got {len(d)}")
    return [d[i] for i in range(1000)]


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


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def evaluate_direct(val_dir, gt_file, pytorch_ckpt_path, classnames_file, batch_size=32, device_id=0):
    start_time = time.time()

    IMAGENET_CLASSNAMES = load_imagenet_classnames(classnames_file)
    print(f"[INFO] Loaded imagenet class names")

    gt_dict = load_ground_truth(gt_file)
    print(f"[INFO] Loaded {len(gt_dict)} ground truth entries")

    image_files = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    print(f"[INFO] Found {len(image_files)} images in {val_dir}")

    # Set NPU device
    device = torch.device(f'npu:{device_id}')
    torch_npu.npu.set_device(device_id)

    print(f"[INFO] Using device: npu:{device_id}")
    print(f"[INFO] Loading model from: {pytorch_ckpt_path}")
    model = AutoModel.from_pretrained(pytorch_ckpt_path, trust_remote_code=False).to(device)
    model.optimize_qkv_for_inference()
    model.eval()
    image_processor = AutoImageProcessor.from_pretrained(pytorch_ckpt_path, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(pytorch_ckpt_path)

    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backbend = tng.get_npu_backend(compiler_config=config)
    model.vision_model = torch.compile(model.vision_model, dynamic=True, fullgraph=True, backend=npu_backbend)
    model.text_model = torch.compile(model.text_model, dynamic=True, fullgraph=True, backend=npu_backbend)

    hypothesis_template = "This is a photo of {}."
    sequences = [hypothesis_template.format(label) for label in IMAGENET_CLASSNAMES]

    with torch.no_grad():
        text_inputs = tokenizer(
            sequences,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    total = 0
    correct = np.zeros(5)
    pbar = tqdm(total=len(image_files), desc="Evaluating", unit="img")

    with torch.no_grad():
        for i in range(0, len(image_files), batch_size):
            batch_fnames = image_files[i:i + batch_size]
            batch_images = []
            batch_keys = []
            for fname in batch_fnames:
                img_key = fname.split('.')[0]
                if img_key not in gt_dict:
                    pbar.update(1)
                    continue
                try:
                    image = load_image(os.path.join(val_dir, fname))
                    batch_images.append(image)
                    batch_keys.append(img_key)
                except Exception:
                    pbar.update(1)
                    continue

            if not batch_images:
                pbar.update(len(batch_fnames))
                continue

            image_inputs = image_processor(
                images=batch_images,
                return_tensors="pt"
            ).to(device)

            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            logits = (image_features @ text_features.T).cpu().numpy()

            for idx in range(len(batch_images)):
                true_label = gt_dict[batch_keys[idx]]
                top5_indices = np.argsort(-logits[idx])[:5]
                hits = [0] * 5
                for j, pred in enumerate(top5_indices):
                    if pred == true_label:
                        hits[j] = 1
                        break
                for k in range(5):
                    correct[k] += hits[k]
                total += 1
                pbar.update(1)

    pbar.close()

    if total == 0:
        print("[ERROR] No valid images processed.")
        return

    accuracy = np.cumsum(correct) / total
    print("\n" + "=" * 50)
    print(f"Total images processed: {total}")
    for i in range(5):
        print(f"Top-{i+1} Accuracy: {accuracy[i] * 100:.2f}%")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Throughput: {total / (time.time() - start_time):.2f} images/sec")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Eval")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation images")
    parser.add_argument("--gt_file", type=str, required=True, help="Ground truth file (filename.jpg label_id)")
    parser.add_argument("--pytorch_ckpt_path", type=str, required=True, help="Path to the PyTorch model checkpoint")
    parser.add_argument("--classnames_file", type=str, required=True, help="Plain-text class names file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device ID (default: 0)")

    args = parser.parse_args()

    for name, path in [
        ("val_dir", args.val_dir),
        ("gt_file", args.gt_file),
        ("pytorch_ckpt_path", args.pytorch_ckpt_path),
        ("classnames_file", args.classnames_file)
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found: {path}")
            sys.exit(1)

    evaluate_direct(
        val_dir=args.val_dir,
        gt_file=args.gt_file,
        pytorch_ckpt_path=args.pytorch_ckpt_path,
        classnames_file=args.classnames_file,
        batch_size=args.batch_size,
        device_id=args.device_id
    )

if __name__ == "__main__":
    main()