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

import argparse
import os
import numpy as np
import torch
import onnx
from torch import nn
from numpy import random

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector.onnx_model import SWIN_BACKBONE, FUSE_MODEL, SELECT_BBOX
from maskrcnn_benchmark.engine.inference import create_positive_dict, create_queries_and_maps, create_queries_and_maps_from_dataset
from maskrcnn_benchmark.modeling.language_backbone import build_language_backbone
from transformers import AutoTokenizer


class lang(nn.Module):
    def __init__(self, cfg):
        super(lang, self).__init__()
        self.language_backbone = build_language_backbone(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
    def forward(self, captions):
        language_dict_features = {}
        tokenized = self.tokenizer.batch_encode_plus(captions,
                                                    max_length=256,
                                                    padding='max_length',
                                                    return_special_tokens_mask=True,
                                                    return_tensors='pt',
                                                    truncation=True).to('cpu')
        input_ids = tokenized.input_ids
        mlm_labels = None
        tokenizer_input = {"input_ids": input_ids,
                        "attention_mask": tokenized.attention_mask}
        language_dict_features = self.language_backbone(tokenizer_input) 
        return language_dict_features


def create_language_dict_features(cfg, weight):
    data_loaders = make_data_loader(cfg, is_train=False)
    data_loader = data_loaders[0]
    dataset = data_loaders[0].dataset
    captions, all_positive_map_label_to_token = create_queries_and_maps_from_dataset(dataset, cfg)
    positive_map=all_positive_map_label_to_token[0]
    lang_model = lang(cfg)
    checkpointer = DetectronCheckpointer(cfg, lang_model)
    _ = checkpointer.load(weight, force=True)
    with torch.no_grad():
        language_dict_features = lang_model(captions)
    return language_dict_features, captions, positive_map


def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config-file",
        default="configs/grounding/e2e_dyhead_SwinT_S_FPN_1x_od_grounding_eval.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        help="pth to model",
        default="glip_tiny_model_o365_goldg.pth"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        "--model_type",
        help="convert model type",
    )

    args = parser.parse_args()    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  
    if args.model_type == "backbone":
        model = SWIN_BACKBONE(cfg)
    elif args.model_type == "rpn_head":
        language_dict_features, captions, positive_map = create_language_dict_features(cfg, args.weight)
        np.save('./rpn_head/positive_map.npy', positive_map)
        model = FUSE_MODEL(cfg, language_dict_features, captions, positive_map)
    elif args.model_type == "select":
        model = SELECT_BBOX(cfg)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(args.weight, force=True)
    iou_types = ("bbox",)
    model.eval()
    if args.model_type == "backbone":
        image = torch.rand(1,3,800,1216)
        dynamic_axes={'images': {2: '-1', 3: '-1'}}
        dummy_input = (image)
        torch.onnx.export(model, dummy_input, './backbone/model/glip_backbone.onnx', input_names=['images'],\
                    output_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'], opset_version=11, dynamic_axes=dynamic_axes) 
    elif args.model_type == "rpn_head":
        feature_1 = torch.rand(1,256,100,152)
        feature_2 = torch.rand(1,256,50,76)
        feature_3 = torch.rand(1,256,25,38)
        feature_4 = torch.rand(1,256,13,19)
        feature_5 = torch.rand(1,256,7,10)
        dynamic_axes={'feature_1': {2: '-1', 3: '-1'}, 'feature_2': {2: '-1', 3: '-1'}, 'feature_3': {2: '-1', 3: '-1'}, \
                      'feature_4': {2: '-1', 3: '-1'}, 'feature_5': {2: '-1', 3: '-1'}}
        dummy_input = (feature_1, feature_2, feature_3, feature_4, feature_5)
        torch.onnx.export(model, dummy_input, './rpn_head/model/glip_rpn_head.onnx', input_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],\
                        output_names=['o1', 'o2', 'o3', 'o4', 'o5', 'o6'], opset_version=11, dynamic_axes=dynamic_axes)
    elif args.model_type == "select":
        input_1 = torch.rand(1,256,100,152)
        input_2 = torch.rand(1,256,50,76)
        input_3 = torch.rand(1,256,25,38)
        input_4 = torch.rand(1,256,13,19)
        input_5 = torch.rand(1,256,7,10)
        input_6 = torch.rand(1,256,768)
        dynamic_axes={'input_1': {2: '-1', 3: '-1'}, 'input_2': {2: '-1', 3: '-1'}, 'input_3': {2: '-1', 3: '-1'}, \
                      'input_4': {2: '-1', 3: '-1'}, 'input_5': {2: '-1', 3: '-1'}}
        dummy_input = (input_1, input_2, input_3, input_4, input_5, input_6) 
        torch.onnx.export(model, dummy_input, './select/model/glip_select.onnx', input_names=['input_1', 'input_2', 'input_3', 'input_4', 'input_5', 'input_6'],\
                          output_names=['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', \
                                        'o14', 'o15', 'o16', 'o17', 'o18', 'o19', 'o20',], \
                          opset_version=11, dynamic_axes=dynamic_axes)   

if __name__ == '__main__':
    main()