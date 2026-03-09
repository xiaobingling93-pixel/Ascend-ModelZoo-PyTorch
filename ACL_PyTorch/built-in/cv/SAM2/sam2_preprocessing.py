# Copyright 2026 Huawei Technologies Co., Ltd
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

from typing import Union, Tuple
from PIL.Image import Image
import torch
import numpy as np
from sam2.utils.transforms import SAM2Transforms


IMAGE_SIZE = 1024


sam2Transforms = SAM2Transforms(
    resolution=IMAGE_SIZE,
    mask_threshold=0.0
)


def encoder_preprocessing(image: Union[np.ndarray, Image]):
    """
    Arguments:
        image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
        with pixel values in [0, 255].
    return:
        input_image (np.ndarray): 1x3xHxW
    """
    if not isinstance(image, Image) and not isinstance(image, np.ndarray):
        raise NotImplementedError("Image format not supported")
    input_image = sam2Transforms(image)
    input_image = input_image[None, ...].numpy() # 1x3xHxW
    return input_image


def decoder_preprocessing(high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray, \
    image_embedding: np.ndarray, image_orig_hw: Tuple[int, int], \
    point_coords: np.ndarray = None, point_labels: np.ndarray = None, box: np.ndarray = None, \
    mask_input: np.ndarray = None):
    """
    Prepare decoder inputs for SAM2 inference.
    This function processes user prompts (points, boxes, masks) and combines them
    with encoder features to create the input format expected by SAM2ImageDecoder.
    Args:
      high_res_feats_0: High-resolution feature map from encoder, shape (1, 32, 256, 256).
      high_res_feats_1: Medium-resolution feature map from encoder, shape (1, 64, 128, 128).
      image_embedding: Low-resolution image embedding from encoder, shape (1, 256, 64, 64).
      image_orig_hw: Original image dimensions as (height, width) tuple, e.g., (1080, 1920).
        Used to denormalize prompt coordinates.
      point_coords: Optional point prompt coordinates, shape (N, 2) or (1, N, 2).
        Coordinates can be normalized [0, 1] or pixel values. If None, no point prompts.
      point_labels: Optional point prompt labels, shape (N,) or (1, N).
        Each value is 1 (foreground), 0 (background), or -1 (ignore). Must be supplied
        if point_coords is supplied.
      box: Optional box prompt, shape (4,) or (1, 4) in [x1, y1, x2, y2] format.
        Coordinates can be normalized [0, 1] or pixel values. If None, no box prompt.
      mask_input: Optional low-resolution mask input from previous iteration,
        shape (1, 256, 256) or (1, 1, 256, 256). Used for iterative refinement.
        If None, will be replaced with zeros.
    Returns:
      decoder_inputs: List of 7 numpy arrays ready for SAM2ImageDecoder inference:
        [0] image_embedding: Shape (1, 256, 64, 64)
        [1] high_res_feats_0: Shape (1, 32, 256, 256)
        [2] high_res_feats_1: Shape (1, 64, 128, 128)
        [3] point_coords: Shape (1, N, 2), normalized to [0, 1]. N=number of points+boxes.
        [4] point_labels: Shape (1, N), values in {0, 1, 2, 3} where:
            0-1: point foreground/background
            2-3: box top-left/bottom-right corners
        [5] mask_input: Shape (1, 1, 256, 256), zeros if no mask provided
        [6] has_mask_input: Shape (1,), binary flag (1.0 if mask provided, 0.0 otherwise)
    Note:
      - Box prompts are converted to 2 point prompts (top-left and bottom-right corners)
    """

    # Transform input prompts
    mask_input, point_coords, point_labels, box = _prep_prompts(
        point_coords=point_coords, point_labels=point_labels, box=box, mask_logits=mask_input, normalize_coords=True, image_orig_hw=image_orig_hw
    )

    if point_coords is not None:
        concat_points = (point_coords, point_labels)
    else:
        concat_points = None

    # Embed prompts
    if box is not None:
        box_coords = box.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int8)
        box_labels = box_labels.repeat(box.size(0), 1)
        # merge "boxes" and "points" into a single "concat_points" input
        if concat_points is not None:
            concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
            concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
            concat_points = (concat_coords, concat_labels)
        else:
            concat_points = (box_coords, box_labels)
            
    if concat_points is not None:
        point_coords, point_labels = concat_points[0].cpu().numpy(), concat_points[1].cpu().numpy()
    else:
        point_coords = np.zeros((1, 1, 2), dtype=np.float32)
        point_labels = np.zeros((1, 1), dtype=np.int8)
        

    if mask_input is None:
        has_mask_input = np.zeros(1, dtype=np.int8)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32) 
    else:
        has_mask_input = np.ones(1, dtype=np.int8)
        
    
    decoder_inputs = [image_embedding, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input]

    return decoder_inputs


def _prep_prompts(
        point_coords, point_labels, box, mask_logits, normalize_coords, image_orig_hw
    ):
    unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
    if point_coords is not None:
        if point_labels is None:
            raise ValueError("point_labels must be supplied if point_coords is supplied.")
        point_coords = torch.as_tensor(
            point_coords, dtype=torch.float
        )
        unnorm_coords = sam2Transforms.transform_coords(
            point_coords, normalize=normalize_coords, orig_hw=image_orig_hw
        )
        labels = torch.as_tensor(point_labels, dtype=torch.int)
        if len(unnorm_coords.shape) == 2:
            unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
    if box is not None:
        box = torch.as_tensor(box, dtype=torch.float)
        unnorm_box = sam2Transforms.transform_boxes(
            box, normalize=normalize_coords, orig_hw=image_orig_hw
        )
    if mask_logits is not None:
        mask_input = torch.as_tensor(
            mask_logits, dtype=torch.float
        )
        if len(mask_input.shape) == 3:
            mask_input = mask_input[None, :, :, :]
        mask_input = mask_input
    return mask_input, unnorm_coords, labels, unnorm_box