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
from typing import List

import auto_optimizer
import numpy as np


class FaAdapter:
    def __init__(self, origin_onnx: auto_optimizer.OnnxGraph):
        self.__graph = origin_onnx
        self.__nodes_to_remove = []
        self.__indices_of_gather_batch = self.__graph.add_initializer(
            'indices_of_gather_batch',
            np.array([0], dtype=np.int64),
        )
        self.__indices_of_gather_seq_len = self.__graph.add_initializer(
            'indices_of_gather_seq_len',
            np.array([1], dtype=np.int64),
        )
        self.__indices_of_axis_to_unsqueeze_for_mask = self.__graph.add_initializer(
            'indices_of_axis_to_unsqueeze_for_mask',
            np.array([1], dtype=np.int64),
        )

    def adapt(self) -> auto_optimizer.OnnxGraph:
        softmaxs = self.__graph.get_nodes('Softmax')
        for softmax in softmaxs:
            self.__adapt_layer(softmax)
        for node in self.__nodes_to_remove:
            self.__graph.remove(node.name, {})
        self.__graph.infer_shape()
        return self.__graph

    def __adapt_layer(self, softmax: auto_optimizer.OnnxNode) -> None:
        # mul transpose
        #   \    /
        #  matmul_1                                  shape_q                       shape_q
        #     |                                       /    \                        /    \
        # reshape_1                  gather_batch gather_seq_len   gather_batch gather_seq_len
        #     |                               |       |    |                |       |    |
        #   add_1      mul transpose add_1 concat_for_mask_shape   add_1 concat_for_mask_shape
        #     |          \    /        \    /                        \    /
        #   add_2  ==>  matmul_1     reshape_1          ==>          reshape_1
        #     |                \     /                                   |
        # reshape_2             add_2                                unsqueeze
        #     |                   |                                      |
        #  softmax             softmax                            flash_attention
        #     |                   |                                      |
        #  matmul_2            matmul_2                              reshape_3
        #     |                   |
        # reshape_3           reshape_3

        layer_name_prefix = '/'.join(softmax.split('/')[:-1])

        matmul_2 = self.__graph.get_next_nodes(softmax.outputs[0])[0]
        reshape_3 = self.__graph.get_next_nodes(matmul_2.outputs[0])[0]
        reshape_2 = self.__graph.get_prev_node(softmax.inputs[0])
        add_2 = self.__graph.get_prev_node(reshape_2.inputs[0])
        add_1 = self.__graph.get_prev_node(add_2.inputs[0])
        reshape_1 = self.__graph.get_prev_node(add_1.inputs[0])
        matmul_1 = self.__graph.get_prev_node(reshape_1.inputs[0])
        mul = self.__graph.get_prev_node(matmul_1.inputs[0])
        transpose = self.__graph.get_prev_node(matmul_1.inputs[1])

        q = mul.inputs[0]
        k = transpose.inputs[0]
        v = matmul_2.inputs[1]
        addend_1 = add_1.inputs[1]
        addend_2 = add_2.inputs[1]

        target_shape = self.__infer_target_shape(layer_index, q)
        add_1.inputs = [addend_1, addend_2]
        reshape_1.inputs = [add_1.outputs[0], target_shape]
        add_2.inputs = [matmul_1.outputs[0], reshape_1.outputs[0]]
        softmax.inputs = [add_2.outputs[0]]

        self.__nodes_to_remove.append(reshape_2)

        # Shape 的输出是 int64，FlashAttentionSoftmaxFp32 的 qSeqLen 和 kvSeqLen 只支持 int32，所以需要用 cast 显式转换
        cast = self.__graph.add_node(
            layer_name_prefix + 'Cast_seq_len',
            'Cast',
            inputs=[layer_name_prefix + 'seq_len'],  # 该输入在 self.__infer_mask_shape 中定义
            outputs=[layer_name_prefix + 'seq_len_int_32'],
            attrs={'to': 6},
        )

        # FlashAttentionSoftmaxFp32 要求 mask 的 shape 为 [b, headsMask, s, s]，其中，headsMask 可以为 1，也可以为 heads
        # 这里 mask 的 shape 为 [b, s, s]，需要 unsqueeze 为 [b, 1, s, s]
        unsqueeze = self.__graph.add_node(
            layer_name_prefix + 'Unsqueeze_mask',
            'Unsqueeze',
            inputs=[reshape_1.outputs[0], self.__indices_of_axis_to_unsqueeze_for_mask.name],
            outputs=[layer_name_prefix + 'unsqueeze_mask_output'],
        )

        # mul 的第二个输入对应 FlashAttentionSoftmaxFp32 的 attr tor
        # 这里 tor 为 0.125，等于 tor 的默认值，不需要显式传参
        flash_attention = self.__graph.add_node(
            layer_name_prefix + 'flash_attention',
            'FlashAttentionSoftmaxFp32',
            inputs=[q, k, v, cast.outputs[0], cast.outputs[0], unsqueeze.outputs[0]],
            outputs=[layer_name_prefix + 'flash_attention_output'],
        )
        reshape_3.inputs[0] = flash_attention.outputs[0]
        self.__nodes_to_remove += [mul, transpose, matmul_1, add_2, softmax, matmul_2]

    def __infer_target_shape(self, layer_name_prefix: str, q: str) -> str:
        shape_q = self.__graph.add_node(
            layer_name_prefix + 'Shape_q',
            'Shape',
            inputs=[q],
            outputs=[layer_name_prefix + 'Shape_of_q'],
        )
        gather_batch = self.__graph.add_node(
            layer_name_prefix + 'Gather_batch',
            'Gather',
            inputs=[shape_q.outputs[0], self.__indices_of_gather_batch.name],
            outputs=[layer_name_prefix + 'batch'],
        )
        gather_seq_len = self.__graph.add_node(
            layer_name_prefix + 'Gather_seq_len',
            'Gather',
            inputs=[shape_q.outputs[0], self.__indices_of_gather_seq_len.name],
            outputs=[layer_name_prefix + 'seq_len'],
        )
        concat_for_mask_shape = self.__graph.add_node(
            layer_name_prefix + 'Concat_for_mask_shape',
            'Concat',
            inputs=[
                gather_batch.outputs[0],
                gather_seq_len.outputs[0],
                gather_seq_len.outputs[0],
            ],
            outputs=[layer_name_prefix + 'mask_shape'],
            attrs={'axis': 0},
        )
        return concat_for_mask_shape.outputs[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Modify the SAM encoder ONNX model to adapt Ascend chips.'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='The path to the original SAM encoder ONNX model.',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='The path to save the adapted SAM encoder ONNX model to.',
    )
    args = parser.parse_args()
    graph = auto_optimizer.OnnxGraph.parse(args.input)
    fa_adapter = FaAdapter(graph)
    graph = fa_adapter.adapt()
    graph.save(args.output)
