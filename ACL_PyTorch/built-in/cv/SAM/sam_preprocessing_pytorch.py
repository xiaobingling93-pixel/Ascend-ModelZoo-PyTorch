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


import cv2
import torch
import numpy as np
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide


IMAGE_SIZE = 1024


def encoder_preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(IMAGE_SIZE)
    image = transform.apply_image(image)
    image = torch.tensor(image)
    h, w, _ = image.shape
    image = F.pad(image, (0, 0, 0, IMAGE_SIZE - w, 0, IMAGE_SIZE - h))
    image = np.array(image, dtype=np.uint8)
    image = image[None, :, :, :]
    return image


def decoder_preprocessing(image_embedding, input_point=None, box=None, image=None):
    coords_list = []
    labels_list = []

    if input_point is not None and len(input_point) > 0:
        input_point = np.array(input_point, dtype=np.float32)
        input_label = np.ones(len(input_point), dtype=np.float32)
        coords_list.append(input_point)
        labels_list.append(input_label)

        coords_list.append(np.array([[0.0, 0.0]], dtype=np.float32))
        labels_list.append(np.array([-1], dtype=np.float32))

    if box is not None:
        box = np.array(box, dtype=np.float32).reshape(2, 2)
        coords_list.append(box)
        labels_list.append(np.array([2, 3], dtype=np.float32))

    onnx_coord = np.concatenate(coords_list, axis=0)[None, :, :]
    onnx_label = np.concatenate(labels_list, axis=0)[None, :].astype(np.float32)

    transform = ResizeLongestSide(IMAGE_SIZE)
    onnx_coord = transform.apply_coords(onnx_coord, image.shape[: 2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    decoder_inputs = [image_embedding, onnx_coord, onnx_label, onnx_mask_input, onnx_has_mask_input]
    return decoder_inputs
