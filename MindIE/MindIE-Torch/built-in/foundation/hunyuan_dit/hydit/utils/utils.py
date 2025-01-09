#!/usr/bin/env python
# coding=utf-8
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


import importlib
import random
import numpy as np
import torch
import PIL
from PIL import Image


def is_npu_available():
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu

    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


def set_seeds_generator(seed, device=None):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return torch.Generator(device).manual_seed(seed)


def randn_tensor(
    shape: tuple,
    generator: torch.Generator = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
):
    """
    A helper function to create random tensors on the desired `device` with the desired `dtype`. When passing
    a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "npu":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def _denormalize(images):
    return (images / 2 + 0.5).clamp(0, 1)


def _pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def _numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def postprocess_pil(image: torch.Tensor):
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"The input image type must be a torch.FloatTensor, but got {type(image)}.")

    image = torch.stack([_denormalize(image[i]) for i in range(image.shape[0])])
    image = _pt_to_numpy(image)

    return _numpy_to_pil(image)