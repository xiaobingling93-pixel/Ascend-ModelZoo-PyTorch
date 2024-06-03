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
from maskrcnn_benchmark.modeling.language_backbone import build_language_backbone

class LANG(nn.Module):
    def __init__(self, cfg):
        super(LANG, self).__init__()
        self.language_backbone = build_language_backbone(cfg)
    def forward(self, input_ids, attention_mask):
        tokenizer_input = {"input_ids": input_ids,
                        "attention_mask": attention_mask}
        language_dict_features = self.language_backbone(tokenizer_input) 
        return language_dict_features['hidden'], language_dict_features['masks']

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
    def __init__(self, cfg):
        super(FUSE_MODEL, self).__init__()
        self.cfg = cfg
        self.rpn = VLDyHeadModule(cfg, onnx_export='rpn_head')


    def forward(self, input1, input2, input3, input4, input5, hidden, mask):
        language_dict_features = {}
        language_dict_features['hidden'] = hidden
        language_dict_features['masks'] = mask
        language_dict_features['embedded'] = None
        visual_features = []
        visual_features.append(input1)
        visual_features.append(input2)
        visual_features.append(input3)
        visual_features.append(input4)
        visual_features.append(input5)
        with torch.no_grad():
            box_regression, centerness, dot_product_logits = self.rpn(None, visual_features, None, language_dict_features, None, None, None)

        return box_regression[0], box_regression[1], box_regression[2], \
               box_regression[3], box_regression[4], centerness[0], centerness[1], centerness[2], centerness[3], centerness[4], \
               dot_product_logits[0], dot_product_logits[1], dot_product_logits[2], dot_product_logits[3], dot_product_logits[4]


