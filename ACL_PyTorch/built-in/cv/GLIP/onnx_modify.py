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

from auto_optimizer import OnnxGraph
import numpy as np
import argparse


class OnnxModel:
    def __init__(self, path):
        self.graph = OnnxGraph.parse(path)

    def remove_useless_node(self):
        # 删除masks后面的unsqueeze算子
        reshape_nodes = self.graph.get_nodes("Reshape")
        for node in reshape_nodes:
            if node.inputs[0] == "masks":
                unsqueeze_nodes = self.graph.get_next_nodes(node.outputs[0])
                # 添加cast算子，输入masks转为fp32
                cast_node = self.graph.add_node("cast_mask", "Cast", attrs={"to": 1})
                self.graph[node.name] = cast_node
                self.graph.remove(unsqueeze_nodes[0].name)
        # 删除softmax之前的reshape
        softmax_node = self.graph.get_nodes("Softmax")
        for node in softmax_node:
            pre_node = self.graph.get_prev_node(node.inputs[0])
            if pre_node.op_type == "Reshape":
                add_node = self.graph.get_prev_node(pre_node.inputs[0])
                reshape_node = self.graph.get_prev_node(add_node.inputs[0])
                self.graph.remove(pre_node.name, {0: 0})
                self.graph.remove(reshape_node.name, {0: 0})
        # 4维转换为3维，删除多余维度
        const_nodes = self.graph.get_nodes("ConstantOfShape")
        for node in const_nodes:
            if not self.graph.get_prev_node(node.inputs[0]):
                self.graph[node.inputs[0]].value = np.array([3], dtype=np.int64)
                where_node = self.graph.get_next_nodes(node.outputs[0])[1]
                shape_node = self.graph[where_node.inputs[2]]
                v = shape_node.value
                self.graph[where_node.inputs[2]].value = v[1:]
        # concat 3维
        concat_nodes = self.graph.get_nodes("Concat")
        for node in concat_nodes:
            if len(node.inputs) == 4 and "Constant" in node.inputs[1] and self.graph[node.inputs[1]].value == [1]:
                node.inputs = [node.inputs[0], node.inputs[2], node.inputs[3]]

    def remove_overflow_node(self):
        node_list = self.graph.get_nodes("ReduceMax")
        for node in node_list:
            output_edges = node.outputs
            next_nodes = self.graph.get_next_nodes(output_edges[0])
            self.graph.remove(node.name)
            self.graph.remove(next_nodes[0].name)

    def save_model(self, output):
        self.graph.save(output)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="onnx model path", default='rpn.onnx')
    parser.add_argument('--output_path', type=str, help="onnx model path after modify", default="modify.onnx")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    G = OnnxModel(args.model_path)
    # 删除softmax防溢出算子reduceMaxD和Sub,删除后不影响精度
    G.remove_overflow_node()
    # 保持输入3维，删除多余扩维算子，删除后不影响精度
    G.remove_useless_node()
    # 保存修改后的onnx模型
    G.save_model(args.output_path)