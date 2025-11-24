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
import os
import multiprocessing
import ast
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer


def load_imagenet_classnames(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        d = ast.literal_eval(f.read())
    if len(d) != 1000:
        raise ValueError(f"Expected 1000 classes, got {len(d)}")
    return [d[i] for i in range(1000)]


def gen_image_input_bin(file_batches, batch, input_dir, image_output_dir, pytorch_ckpt_path):
    processor = AutoImageProcessor.from_pretrained(pytorch_ckpt_path, use_fast=False)
    for file_name in file_batches[batch]:
        with Image.open(os.path.join(input_dir, file_name)) as pilimg:
            pilimg = pilimg.convert("RGB")
            img_numpy = processor(images=pilimg, return_tensors="np")
            img_numpy["pixel_values"].tofile(os.path.join(image_output_dir, file_name.split('.')[0] + ".bin"))


def gen_text_input_bin(text_output_dir, pytorch_ckpt_path, classnames_file):
    processor = AutoTokenizer.from_pretrained(pytorch_ckpt_path)
    candidate_labels = load_imagenet_classnames(classnames_file)
    hypothesis_template = "This is a photo of {}."
    sequences = [hypothesis_template.format(x) for x in candidate_labels]
    text_numpy = processor(sequences, padding="max_length", max_length=64, truncation=True, return_tensors="np")
    text_numpy["input_ids"].tofile(os.path.join(text_output_dir, "IMAGENET_CLASSNAMES_10000.bin"))


def preprocess(input_dir, image_output_dir, text_output_dir, pytorch_ckpt_path, classnames_file):
    file_names = os.listdir(input_dir)
    total_nums = len(file_names)
    batch_size = max(1, total_nums // 10)
    file_batches = [file_names[i:i + batch_size] for i in range(0, total_nums, batch_size) if file_names[i:i + batch_size]]
    pbar = tqdm(total=len(file_batches))
    pbar.set_description("Preprocessing")
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_image_input_bin, args=(file_batches, batch, input_dir, image_output_dir, pytorch_ckpt_path), callback=lambda *args: pbar.update(),
        error_callback=lambda e: print(f"Process error: {e}"))
    thread_pool.close()
    thread_pool.join()
    
    gen_text_input_bin(text_output_dir, pytorch_ckpt_path, classnames_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image and text preprocessing script")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of the image dataset")
    parser.add_argument("--image_save_dir", type=str, required=True, help="Directory to save processed image")
    parser.add_argument("--text_save_dir", type=str, required=True, help="Directory to save processed text")
    parser.add_argument("--pytorch_ckpt_path", type=str, required=True, help="Path to the PyTorch model checkpoint")
    parser.add_argument("--classnames_file", type=str, required=True, help="Plain-text class names file")

    args = parser.parse_args()

    if not os.path.isdir(args.image_save_dir):
        os.makedirs(os.path.realpath(args.image_save_dir))
    if not os.path.isdir(args.text_save_dir):
        os.makedirs(os.path.realpath(args.text_save_dir))

    preprocess(args.data_dir, args.image_save_dir, args.text_save_dir, args.pytorch_ckpt_path, args.classnames_file)