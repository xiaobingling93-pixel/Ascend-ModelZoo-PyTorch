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
from auto_optimizer import OnnxGraph

from mmseg.apis import init_model
import mmseg_custom  # noqa: F401,F403


def delete_domain(graph):
    for node in graph.nodes:
        if node.domain != '':
            node.domain = ''
    while len(graph.opset_imports) > 1:
        graph.opset_imports.pop(1)


def main(cfg_path, ckpt_path, export_path):
    export_dir = os.path.dirname(export_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    model = init_model(cfg_path, checkpoint=None, device='cpu')
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    load_state_dict(model, checkpoint['state_dict'], strict=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 896, 1216)

    torch.onnx.export(model, dummy_input, export_path,
                      export_params=True, opset_version=16, verbose=False,
                      dynamic_axes={
                          "input": {0: "batch", 2: "height", 3: "width"},
                          "output": {0: "batch", 2: "height", 3: "width"}
                      },
                      input_names=['input'],
                      output_names=['output'],
                      keep_initializers_as_inputs=False)
    print('successfully export onnx')
    onnx_graph = OnnxGraph.parse(export_path)
    delete_domain(onnx_graph)
    resize_list = onnx_graph.get_nodes('Resize')
    for re in resize_list:
        re.attrs['coordinate_transformation_mode'] = 'pytorch_half_pixel'
    onnx_graph.save(export_path)
    print('successfully delete domain')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint file path')
    parser.add_argument('--export', type=str, required=True, help='export ONNX file path')
    args = parser.parse_args()
    main(args.config, args.ckpt, args.export)
