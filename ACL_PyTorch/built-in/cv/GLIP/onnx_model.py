# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
from torch import nn

from ..backbone import build_backbone
from ..rpn.vldyhead import VLDyHeadModule


class SWIN_BACKBONE(nn.Module):
    def __init__(self, cfg):
        super(SWIN_BACKBONE, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)

    def forward(self, images):
        with torch.no_grad():
            visual_features = self.backbone(images)
        return visual_features[0], visual_features[1], visual_features[2], visual_features[3], visual_features[4]

class FUSE_MODEL(nn.Module):
    def __init__(self, cfg, language_dict_features, captions, positive_map):
        super(FUSE_MODEL, self).__init__()
        self.cfg = cfg
        self.rpn = VLDyHeadModule(cfg, onnx_export='rpn_head')
        self.language_dict_features = language_dict_features
        self.captions = captions
        self.positive_map = positive_map

    def forward(self, input1, input2, input3, input4, input5):
        visual_features = []
        visual_features.append(input1)
        visual_features.append(input2)
        visual_features.append(input3)
        visual_features.append(input4)
        visual_features.append(input5)
        with torch.no_grad():            
            o1, o2, o3, o4, o5, o6 = self.rpn(None, visual_features, None, self.language_dict_features, self.positive_map, self.captions, None)

        return o1, o2, o3, o4, o5, o6

class SELECT_BBOX(nn.Module):
    def __init__(self, cfg):
        super(SELECT_BBOX, self).__init__()
        self.cfg = cfg
        self.rpn = VLDyHeadModule(cfg, onnx_export='select')

    def forward(self, input1, input2, input3, input4, input5, input6):
        visual_features = [input1, input2, input3, input4, input5, input6]
        with torch.no_grad():
            box_cls, box_regression, centerness, dot_product_logits = self.rpn(None, visual_features, None, None, None, None, None)

        return box_cls[0], box_cls[1], box_cls[2], box_cls[3], box_cls[4], box_regression[0], box_regression[1], box_regression[2], \
               box_regression[3], box_regression[4], centerness[0], centerness[1], centerness[2], centerness[3], centerness[4], \
               dot_product_logits[0], dot_product_logits[1], dot_product_logits[2], dot_product_logits[3], dot_product_logits[4]
