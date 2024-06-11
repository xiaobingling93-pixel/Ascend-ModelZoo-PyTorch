# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import sys
import numpy as np
from copy import deepcopy
from auto_optimizer import OnnxGraph

FP16_MAX_VALUE = 65504
FP16_MIN_VALUE = -65504

def delete_domain(graph):
    for node in graph.nodes:
        if node.domian != '':
            node.domain = ''
    while len(graph.opset_imports) > 1:
        graph.opset_imports.pop(1)

def fix_mul(graph):
    initializer_list = graph.initializers
    mul_input = []
    for mul_node in graph.get_nodes('Mul'):
        for input_name in mul_node.inputs:
            mul_input.append(input_name)
    for initializer in initializer_list:
        if initializer.name in mul_input:
            fixed_value = deepcopy(initializer.value)
            value_mask_pos = (fixed_value > FP16_MAX_VALUE)
            value_mask_neg = (fixed_value < FP16_MIN_VALUE)
            if np.sum(value_mask_pos) > 0 or np.sum(value_mask_neg) > 0:
                print(f"Fix value node: {initializer}")
                fixed_value[value_mask_pos] = FP16_MAX_VALUE
                fixed_value[value_mask_neg] = FP16_MIN_VALUE
                initializer.value = fixed_value

if __name__ == '__main__':
    input_path = sys.argv[1]
    save_path = sys.argv[2]
    onnx_graph = OnnxGraph.parse(input_path)
    fix_mul(onnx_graph)
    delete_domain(onnx_graph)
    onnx_graph.save(save_path)

