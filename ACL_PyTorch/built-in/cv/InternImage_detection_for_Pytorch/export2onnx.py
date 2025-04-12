# Copyright 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import argparse

import mmcv
import torch
from mmengine.runner import load_checkpoint, load_state_dict
from mmengine import Config
from mmdet.apis.inference import init_detector
import numpy as np
from auto_optimizer import OnnxGraph

import mmdet_custom  # noqa: F401,F403


def delete_domain(graph):
    for node in graph.nodes:
        if node.domain != '':
            node.domain = ''
    while len(graph.opset_imports) > 1:
        graph.opset_imports.pop(1)


def main(cfg_path, ckpt_path, data_dir, export_path, img_shape_path="./img_shape"):
    export_dir = os.path.dirname(export_path)
    os.makedirs(export_dir, exist_ok=True)

    model = init_detector(cfg_path, checkpoint=None, device='cpu')
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    load_state_dict(model, checkpoint['state_dict'], strict=False)
    model.eval()

    file_path = os.path.join(data_dir, os.listdir(data_dir)[0])
    data_input = torch.from_numpy(np.load(file_path))
    img_shape_file_path = os.path.join(img_shape_path, os.path.basename(file_path))
    img_shape = torch.from_numpy(np.load(img_shape_file_path))

    torch.onnx.export(model, (data_input, img_shape), export_path,
                      opset_version=16, verbose=False,
                      input_names=['data', 'img_shape'],
                      output_names=['cls_scores', 'bboxes', 'feature_map', 'rois'],
                      keep_initializers_as_inputs=False)
    print('successfully export onnx')
    onnx_graph = OnnxGraph.parse(export_path)
    delete_domain(onnx_graph)
    onnx_graph.save(export_path)
    print('successfully delete domain')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('--config', type=str, help='config file path', required=True)
    parser.add_argument('--ckpt', type=str, help='checkpoint file path', required=True)
    parser.add_argument('--data', type=str, help='directory of preprocessed data', required=True)
    parser.add_argument('--img_shape_path', type=str, default="./img_shape", help='directory that saves the img shape')
    parser.add_argument('--export', type=str, help='ONNX file path to be exported', required=True)
    args = parser.parse_args()

    main(args.config, args.ckpt, args.data, args.export, args.img_shape_path)
