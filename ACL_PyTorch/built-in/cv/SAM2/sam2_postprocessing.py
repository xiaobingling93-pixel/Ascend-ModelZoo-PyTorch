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

import torch
from torch.nn import functional as F


def decoder_postprocessing(masks, origin_h, origin_w):
    """
    This function resizes masks output from SAM2ImageDecoder back 
      to the original input image dimensions using bilinear interpolation.
    Arguments:
        masks (np.ndarray): 1 x 1x 256 x 256
    return:
        masks (np.ndarray): 1 x 1 x H x W
    """
    
    mask_tensor = torch.tensor(masks)
    mask_tensor = F.interpolate(
        mask_tensor,
        size=(origin_h, origin_w),
        mode="bilinear",
        align_corners=False
    )
    return mask_tensor.numpy()