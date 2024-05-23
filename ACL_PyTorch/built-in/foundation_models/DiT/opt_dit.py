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


import sys
import argparse
import math
import numpy as np
from argparse import Namespace
from auto_optimizer import OnnxGraph, OnnxNode


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help='path to model'
    )
    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='save dir for onnx model'
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='batchsize of dit model'
    )
    parser.add_argument(
        '--FA_soc',
        default='none', 
        type=str, 
        help='Type of FA operator',
        choices=['Duo', 'A2', 'none']
    )

    return parser.parse_args()


def node_check(node) -> tuple:
    idx = 0
    if isinstance(node, str):
        op_type = node
    elif isinstance(node, tuple):
        op_type, idx = node
    else:
        raise TypeError(f"Invalid preorder type: {type(p)}!")

    return idx, op_type

def pattern_select(
    graph: OnnxGraph,
    candidate_nodes: list, 
    preorders: list = None, 
    successors: list = None
) -> list:
    ret = []
    preorders = preorders or []
    successors = successors or []

    for node in candidate_nodes:
        pattern_check = True
        current_node = node
        # check if the preceding node is the ecpected structure
        for p in preorders[::-1]:
            input_idx, op_type = node_check(p)
            current_node = graph.get_prev_node(current_node.inputs[input_idx])
            if not current_node or current_node.op_type != op_type:
                pattern_check = False
                break

        if not pattern_check:
            continue

        current_node = node
        #Check if the post node is the expected structure
        for s in successors:
            output_idx, op_type = node_check(s)
            next_nodes = graph.get_next_nodes(current_node.outputs[output_idx])
            pattern_check = False
            for next_node in next_nodes:
                if next_node.op_type == op_type:
                    current_node = next_node
                    pattern_check = True
                    break

            if not pattern_check:
                break

        if pattern_check:
            ret.append(node)

    return ret


def get_attention_matmul_nodes(graph: OnnxGraph) -> list:
    # Pattern: Mul -> [MatMul] -> SoftMax
    all_matmul_nodes = graph.get_nodes("MatMul")
    return pattern_select(graph, all_matmul_nodes, ["Mul"], ["Softmax"])


def get_attention_split_nodes(graph: OnnxGraph) -> list:
    # Pattern : Transpose -> [Split] -> Squeeze
    all_split_nodes = graph.get_nodes("Split")
    return pattern_select(graph, all_split_nodes, ["Transpose"], ["Squeeze"])


def get_attention_reshape_nodes(graph: OnnxGraph) -> list:
    # Pattern : Transpose -> [Reshape] -> MatMul
    all_reshape_nodes = graph.get_nodes("Reshape")
    return pattern_select(graph, all_reshape_nodes, ["Transpose"], ["MatMul"])


def cal_padding_shape(
    graph: OnnxGraph,
    split_node: OnnxNode,
    reshape_node: OnnxNode,
    const_nodes: list
) -> None:
    transpose_node1 = graph.get_prev_node(split_node.inputs[0])
    transpose_node2 = graph.get_prev_node(reshape_node.inputs[0])

    graph.connect_node(
        graph.add_node(split_node.name.replace("Split", "Concat"), "Concat", attrs={"axis":4}),
        [transpose_node1.name, const_nodes[-1].name],
        [split_node.name]
    )

    graph.connect_node(
        graph.add_node(split_node.name.replace("Split", "Slice"), "Slice"),
        [transpose_node2.name, const_nodes[0].name, const_nodes[1].name, const_nodes[2].name],
        [reshape_node.name]
    )


def flash_attention_op(
    graph: OnnxGraph,
    matmul_node:OnnxNode,
    fa_name: str
) -> None:
    softmax_node = graph.get_next_nodes(matmul_node.outputs[0])[0]
    matmul_node1 = graph.get_next_nodes(softmax_node.outputs[0])[0]
    transpose_node = graph.get_prev_node(graph.get_prev_node(matmul_node.inputs[1]).inputs[0])

    fa_node = graph.add_node(softmax_node.name.replace("Softmax", fa_name), fa_name)
    fa_node.inputs = [
        matmul_node.inputs[0],
        matmul_node.inputs[1],
        matmul_node1.inputs[1]
    ]
    fa_node.outputs = matmul_node1.outputs

    graph.remove(softmax_node.name, {})
    graph.remove(matmul_node1.name, {})
    graph.remove(matmul_node.name, {})
    graph.remove(transpose_node.name)


def adapt_for_flashattention(graph: OnnxGraph, soc_type: str) -> None:
    """
    pattern:
            /         |          \                                          /        |          \
      Squeeze_v   Squeeze_q   Squeeze_k                              Squeeze_v   Squeeze_q   Squeeze_k
            |         |          |                                         |         |          |
            |         |         Mul                                        |         |          |
            |         |         /                 adapt                    |         |          |
            |        Mul   Transpose_k           =======>                  |        Mul        Mul
            |          \      /                                             \        |         /
             \         MatMul                                                \       |        /
              \           |                                                   FlashAttentionTik
               \       SoftMax                                                       |
                \        /                                                       Transpose
                  MatMul                                                             |
                    |                                                             Reshape
                 Transpose                                                        
                    |                                                               
                  Reshape                                                          

    """
    fa_name = "FlashAttentionTik" if soc_type == "Duo" else "UnpadFlashAttentionMix"
    matmuls = get_attention_matmul_nodes(graph)
    splits = get_attention_split_nodes(graph)
    reshapes = get_attention_reshape_nodes(graph)

    const_nodes = []
    padding_shape = graph.get_value_info(splits[0].inputs[0]).shape
    const_nodes.append(graph.add_initializer("blocks.0/attn/Constant_for_Slice_1", np.array([0])))
    const_nodes.append(graph.add_initializer("blocks.0/attn/Constant_for_Slice_2", np.array([[padding_shape[-1]]])))
    const_nodes.append(graph.add_initializer("blocks.0/attn/Constant_for_Slice_3", np.array([3])))
    padding_shape[-1], padding_shape[1] = math.ceil(padding_shape[-1] / 16) * 16 - padding_shape[-1], args.batch_size
    const_nodes.append(
        graph.add_initializer("blocks.0/attn/Constant_for_Concat", np.zeros(padding_shape, dtype=np.float32))
    )

    # padding shape [bs, 1024, 3, 16, 72] to [bs, 1024, 3, 16, 80]
    for i, node in enumerate(splits):
        cal_padding_shape(graph, node, reshapes[i], const_nodes)

    for node in matmuls:
        flash_attention_op(graph, node, fa_name)


def main() -> None:
    graph = OnnxGraph.parse(args.model)

    if args.FA_soc == "Duo":
        adapt_for_flashattention(graph, args.FA_soc)
    
    graph.update_map()
    graph.remove_unused_nodes()
    graph.infer_shape()
    graph.save(args.output)


if __name__ == "__main__":
    args = parse_arguments()
    main()
