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

import argparse

import numpy as np
from auto_optimizer import OnnxGraph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        type=str, 
        default='blip_models/text_decoder.onnx',
        help='Path of ONNX model to modify.',
        )
    parser.add_argument(
        '--new_model', 
        type=str, 
        default='blip_models/text_decoder_md.onnx',
        help='Path to save modified ONNX model.',
        )
    return parser.parse_args()


def remove_output(model):
    outputs = [output.name for output in model.outputs]
    for output in outputs:
        if output != 'logits':
            model.remove(output)
    return


def remove_mask(model):
    next_nodes = model.get_next_nodes('encoder_attention_mask')
    del_nodes = []
    while len(next_nodes) == 1:
        del_nodes.extend(next_nodes)
        next_nodes = model.get_next_nodes(next_nodes[0].outputs[0])

    for node in del_nodes:
        model.remove(node.name, {})
    for node in next_nodes:
        model.remove(node.name)
    return


def main(args):
    model = OnnxGraph.parse(args.model)
    remove_output(model)
    remove_mask(model)
    model.update_map()
    model.remove_unused_nodes()
    model.save(args.new_model)


if __name__ == "__main__":
    main(parse_args())
