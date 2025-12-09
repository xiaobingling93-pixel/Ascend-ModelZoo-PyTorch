# Copyright © 2021 - 2025. Huawei Technologies Co., Ltd. All Rights Reserved.
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

import argparse
import os
import math
import multiprocessing


import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def resize_image(image):
    origin_height, origin_width, _ = image.shape
    height = 800
    width = origin_width * height / origin_height
    N = math.ceil(width / 32)
    width = N * 32
    image = cv2.resize(image, (width, height))
    return image


def calculate_dataset_mean(src_path):
    """Calculate mean BGR values of dataset"""
    files = os.listdir(src_path)
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in files if os.path.splitext(f)[-1].lower() in img_extensions]
    total_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)  

    print(f"Calculating mean values of BGR of dataset, path{src_path}...")
    for file in tqdm(files):
        img_path = os.path.join(src_path, file)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        if image is None:
            print(f"Warning：Can not read image of {img_path}, skip.")
            continue

        image = resize_image(image)
        channel_sum += image.sum(axis=(0, 1))
        total_pixels += image.shape[0] * image.shape[1]

    bgr_mean = channel_sum / total_pixels
    print(f"Mean BGR values of dataset: {bgr_mean.tolist()}")
    return bgr_mean.tolist()


def gen_input_npy(file_batches, batch, save_path, bgr_mean_values):
    i = 0
    shapes = set()
    

    for file in file_batches[batch]:
        i = i + 1
        orig_path = os.path.join(flags.image_src_path, file)
        image = cv2.imread(orig_path, cv2.IMREAD_COLOR).astype('float32')

        # Handle the issue of incorrect rotation of 651.jpg in the official dataset
        if file == "img651.jpg":
            output_path = "./img651.jpg"
            if not os.path.exists(output_path):  
                image = ndimage.rotate(image, 90, cval=255)
                cv2.imwrite(output_path, image)
                cv2.imwrite(orig_path, image)
                print("The wrong file image651.jpg has been rotated")
            else:
                print(f"File {output_path} exists, And it does not need to be rotated twice.")

        image = resize_image(image)
        image -= np.array(bgr_mean_values)  # mean values
        image = image / 255.
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, :]  # CHW ---> NCHW
        np.save(os.path.join(save_path, file.split('.')[0] + ".npy"), image)
        shapes.add(image.shape)
    res = [item[3] for item in shapes]
    print(sorted(res))


def preprocess(flags):
    src_path = flags.image_src_path
    save_path = flags.npu_file_path
    bgr_mean_values = calculate_dataset_mean(src_path)
    files = os.listdir(src_path)
    file_batches = [files[i:i + 20] for i in range(0, 5000, 20) if files[i:i + 20] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_npy, args=(file_batches, batch, save_path, bgr_mean_values))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure npy files generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess of db pytorch')
    parser.add_argument('--image_src_path', default="./datasets/icdar2015/test_images", help='images of dataset')
    parser.add_argument('--npu_file_path', default="./icdar2015_npy/", help='npy data')
    flags = parser.parse_args()
    if not os.path.isdir(flags.npu_file_path):
        os.makedirs(os.path.realpath(flags.npu_file_path))
    preprocess(flags)