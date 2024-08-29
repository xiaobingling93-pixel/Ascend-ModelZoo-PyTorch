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
from typing import List

import numpy as np
from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.interface.base_node import Initializer


def get_config(graph):
    input_ph = graph.inputs[0]
    bs, seq_len = input_ph.shape[0], input_ph.shape[1]
    return bs, seq_len


def fix_attention_lnqkv(graph, qkv_start_node):
    # change transpose node
    seen: List[List[int]] = []
    next_nodes = graph.get_next_nodes(qkv_start_node.outputs[0])
    matmul_nodes = [n for n in next_nodes if n.op_type == "MatMul"]
    for idx in range(3):
        matmul_node = matmul_nodes[idx]
        add_node = graph.get_next_nodes(matmul_node.outputs[0])[0]
        reshape_node = graph.get_next_nodes(add_node.outputs[0])[0]
        transpose_node = graph.get_next_nodes(reshape_node.outputs[0])[0]
        perm: List[int] = transpose_node.attrs.get('perm', [1])
        if perm in seen:
            seen.remove(perm)
            query_perm = perm
        else:
            seen.append(perm)
            key_perm = perm
            key_transpose = transpose_node
        
    # [0, 2, 3, 1] -> [0, 2, 1, 3] [0, 1, 3, 2]
    key_transpose.attrs["perm"] = query_perm
    new_perm = [query_perm.index(key_perm[i]) for i in range(len(key_perm))]
    new_transpose = graph.add_node(
        name=f"{key_transpose.name}_after",
        op_type="Transpose",
        attrs={"perm": new_perm}
    )
    graph.insert_node(key_transpose.name, new_transpose, mode="after")


def fix_attention_score(graph, softmax_node, bs, seq_len):
    # fix reshape node 
    matmul_node = graph.get_next_nodes(softmax_node.outputs[0])[0]
    transpose_node = graph.get_next_nodes(matmul_node.outputs[0])[0]
    add_node = graph.get_prev_node(softmax_node.inputs[0])
    prev_node = graph.get_prev_node(add_node.inputs[0])
    if prev_node.op_type == "Div":
        div_node = prev_node
        refer_index = 0
    else:
        div_node = graph.get_prev_node(add_node.inputs[1])
        refer_index = 1
    div_init = graph.get_node(div_node.inputs[0], node_type=Initializer) or \
            graph.get_node(div_node.inputs[1], node_type=Initializer)
    mul_node = graph.add_node(
        f"bert_Mul_before_{add_node.name}",
        "Mul",
    )
    mul_init_value = np.array(1/div_init.value, dtype="float32")
    mul_init = graph.add_initializer(
        f"{mul_node.name}_value",
        mul_init_value
    )
    graph.insert_node(add_node.name, mul_node, refer_index=refer_index, mode="before")
    mul_node.inputs.append(mul_init.name)
    graph.remove(div_node.name)


def main(graph):
    # get config
    bs, seq_len = get_config(graph)
    # fix_lnqkv
    add_nodes = graph.get_nodes("Add")
    gather_node = graph.get_nodes("Gather")[0]
    for add_node in add_nodes:
        if len(graph.get_next_nodes(add_node.outputs[0])) == 4:
            fix_attention_lnqkv(graph, add_node)
    # fix_attentionscore
    softmax_nodes = graph.get_nodes("Softmax")
    for softmax_node in softmax_nodes:
        fix_attention_score(graph, softmax_node, bs, seq_len)
    # add expand node
    expand_node = graph.add_node(
        f"Expand_Mask",
        "Expand"
    )
    expand_init = graph.add_initializer(
        f"expand_value",
        np.array([bs, 1, seq_len, seq_len], dtype="int64")
    )
    s_node = softmax_nodes[0]
    a_node = graph.get_prev_node(s_node.inputs[0])
    m_node = graph.get_prev_node(a_node.inputs[1])
    expand_node.inputs=["mul_out", "expand_value"]
    expand_node.outputs=[m_node.outputs[0]]
    m_node.outputs=["mul_out"]
    # insert last reshape to recover shape
    last_add = graph.get_nodes(op_type="Add")[-1]
    last_reshape = graph.add_node(
        "last_reshape",
        "Reshape"
    )
    reshape_init = graph.add_initializer(
        f"{last_reshape.name}_value",
        np.array([bs, seq_len, HIDDEN_NUM], dtype="int64")
    )
    graph.insert_node(last_add.name, last_reshape, mode="after")
    last_reshape.inputs.append(reshape_init.name)


if __name__=="__main__":
    HIDDEN_NUM=768
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    onnx_graph = OnnxGraph.parse(input_model)
    main(onnx_graph)
    onnx_graph.infer_shape()
    onnx_graph.save(output_model)
