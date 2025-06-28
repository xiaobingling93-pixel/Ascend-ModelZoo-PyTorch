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

import json
import logging
import multiprocessing
import random
import os
import importlib
import time
from dataclasses import dataclass
from multiprocessing import Manager, shared_memory
from threading import Timer

import numpy as np
import torch
import torchvision.io as io
import torch.distributed as dist

import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MASK_DEFAULT = ["0", "0", "0", "0", "1", "0"]
MAX_SHM_SIZE = 10**9
OPENAI_CLIENT = None
REFINE_PROMPTS = None
REFINE_PROMPTS_TEMPLATE = """
You need to refine user's input prompt. The user's input prompt is used for video generation task. You need to refine the user's prompt to make it more suitable for the task. Here are some examples of refined prompts:
{}

The refined prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The refined prompt should be in English.
"""
RANDOM_PROMPTS = None
RANDOM_PROMPTS_TEMPLATE = """
You need to generate one input prompt for video generation task. The prompt should be suitable for the task. Here are some examples of refined prompts:
{}

The prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The prompt should be in English.
"""
REFINE_EXAMPLE = [
    "a close - up shot of a woman standing in a room with a white wall and a plant on the left side."
    "the woman has curly hair and is wearing a green tank top."
    "she is looking to the side with a neutral expression on her face."
    "the lighting in the room is soft and appears to be natural, coming from the left side of the frame."
    "the focus is on the woman, with the background being out of focus."
    "there are no texts or other objects in the video.the style of the video is a simple,"
    " candid portrait with a shallow depth of field.",
    "a serene scene of a pond filled with water lilies.the water is a deep blue, "
    "providing a striking contrast to the pink and white flowers that float on its surface."
    "the flowers, in full bloom, are the main focus of the video."
    "they are scattered across the pond, with some closer to the camera and others further away, "
    "creating a sense of depth.the pond is surrounded by lush greenery, adding a touch of nature to the scene."
    "the video is taken from a low angle, looking up at the flowers, "
    "which gives a unique perspective and emphasizes their beauty."
    "the overall composition of the video suggests a peaceful and tranquil setting, likely a garden or a park.",
    "a professional setting where a woman is presenting a slide from a presentation."
    "she is standing in front of a projector screen, which displays a bar chart."
    "the chart is colorful, with bars of different heights, indicating some sort of data comparison."
    "the woman is holding a pointer, which she uses to highlight specific parts of the chart."
    "she is dressed in a white blouse and black pants, and her hair is styled in a bun."
    "the room has a modern design, with a sleek black floor and a white ceiling."
    "the lighting is bright, illuminating the woman and the projector screen."
    "the focus of the image is on the woman and the projector screen, with the background being out of focus."
    "there are no texts visible in the image."
    "the relative positions of the objects suggest that the woman is the main subject of the image, "
    "and the projector screen is the object of her attention."
    "the image does not provide any information about the content of the presentation or the context of the meeting."
]
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.1
TOP_P = 0.95
TOP_K = 100
SEED = 10
REPETITION_PENALTY = 1.03

TIMEOUT_T = 600


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


def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.

    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def path_check(path: str):
    """
    check path
    param: path
    return: data real path after check
    """
    if os.path.islink(path) or path is None:
        raise RuntimeError("The path should not be None or a symbolic link file.")
    path = os.path.realpath(path)
    if not check_owner(path):
        raise RuntimeError("The path is not owned by current user or root.")
    if not os.path.exists(path):
        raise RuntimeError("The path does not exist.")
    return path


def check_owner(path: str):
    """
    check the path owner
    param: the input path
    return: whether the path owner is current user or not
    """
    path_stat = os.stat(path)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    user_check = path_owner == os.getuid() and path_owner == os.geteuid()
    return path_owner == 0 or path_gid in os.getgroups() or user_check
