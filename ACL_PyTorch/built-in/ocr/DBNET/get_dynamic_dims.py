# Copyright © 2021 - 2025. Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import math
import argparse

import cv2
from tqdm import tqdm


def get_dynamic_widths(image_dir, fixed_height=800):
    """Get all widths of scaled images."""
    width_set = set()
    
    for img_name in tqdm(os.listdir(image_dir), desc="Divining target widths ... "):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img_path = os.path.join(image_dir, img_name)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
            if img is None:
                print(f"Skip invalid images: {img_name}")
                continue
            origin_h, origin_w = img.shape[:2]
            scale = fixed_height / origin_h
            scaled_w = origin_w * scale
            N = math.ceil(scaled_w / 32)
            final_w = N * 32
            width_set.add(final_w)
        except Exception as e:
            print(f"Fail to process {img_name} : {e}")
            continue
    
    sorted_widths = sorted(list(width_set))
    return sorted_widths


def generate_atc_dynamic_dims(image_dir, fixed_height=800):
    """Generate strings used for ATC command."""
    widths = get_dynamic_widths(image_dir, fixed_height)
    dynamic_dims_str = ";".join(map(str, widths))
    return dynamic_dims_str, widths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', 
                        default="./datasets/total_text/test_images", 
                        help='dataset path')
    parser.add_argument('--fixed_height', 
                        type=int, 
                        default=800, 
                        help='fixed height')
    args = parser.parse_args()
    
    dynamic_dims_str, widths = generate_atc_dynamic_dims(args.image_dir, args.fixed_height)
    
    print("=" * 50)
    print("\nThe value of the dynamic dims parameter available for the ATC command:")
    print(dynamic_dims_str)
    
