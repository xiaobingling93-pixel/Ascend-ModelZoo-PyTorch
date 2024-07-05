# Copyright 2024 Huawei Technologies Co., Ltd
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

import auto_optimizer
from auto_optimizer.graph_optimizer import optimizer
from auto_optimizer.graph_refactor.interface import base_graph
from auto_optimizer.graph_refactor.interface import base_node
from auto_optimizer.graph_refactor.onnx import node as onnx_node
from auto_optimizer.pattern import knowledges
from auto_optimizer.pattern import knowledge_factory
from auto_optimizer.pattern import matcher
from auto_optimizer.pattern import pattern as pattern_module
from auto_optimizer.pattern import utils as pattern_utils
from auto_optimizer.pattern.knowledges import knowledge_base
import numpy as np


def _remove_edge(pattern: pattern_module.Pattern, source_name: str, target_name: str) -> None:
    source_node = pattern._nodes.get(source_name)
    target_node = pattern._nodes.get(target_name)
    source_node._outputs.remove(target_node)
    target_node._inputs.remove(source_node)


def _replace_input(next_node: base_node.BaseNode, old_input: str, new_input: str) -> None:
    for i in range(len(next_node.inputs)):
        if next_node.inputs[i] == old_input:
            next_node.inputs[i] = new_input


class IsSecondInputOfMatMul(pattern_module.MatchBase):
    def match(self, node: base_node.BaseNode, graph: base_graph.BaseGraph) -> bool:
        next_node = graph.get_next_nodes(node.outputs[0])[0]
        return next_node.op_type == 'MatMul' and next_node.inputs[1] == node.outputs[0]


class KnowledgeBaseWithPostProcess(knowledge_base.KnowledgeBase):
    def post_process(self, graph: base_graph.BaseGraph) -> bool:
        graph.remove_unused_nodes()
        return True


class KnowledgeOptimizer:
    _knowledge_class: knowledge_base.KnowledgeBase

    def __init__(self, graph: base_graph.BaseGraph, match_result: matcher.MatchResult):
        self._graph = graph


@knowledge_factory.KnowledgeFactory.register()
class KnowledgeMoveMulOrDivBeforeMatmul(KnowledgeBaseWithPostProcess):
    """
     MatMul             Mul/Div
       |        ==>        |
    Mul/Div              MatMul
    """
    _PATTERN_MATMUL = 'MatMul'
    _PATTERN_MUL_OR_DIV = 'Mul_or_Div'

    def __init__(self):
        super().__init__()
        self._register_apply_funcs(self._define_pattern(), [self._apply])

    def _define_pattern(self):
        return pattern_module.Pattern() \
            .add_node(self.__class__._PATTERN_MATMUL, ['MatMul'], [pattern_utils.NextNodeCount(1)]) \
            .add_node(self.__class__._PATTERN_MUL_OR_DIV, ['Mul', 'Div'], [pattern_utils.NextNodeCount(1)]) \
            .add_edge(self.__class__._PATTERN_MATMUL, self.__class__._PATTERN_MUL_OR_DIV)

    def _apply(self, graph: base_graph.BaseGraph, match_result: matcher.MatchResult) -> bool:
        nodes = match_result.node_dicts[0]
        matmul = graph[nodes[self.__class__._PATTERN_MATMUL][0].name]
        mul_or_div = graph[nodes[self.__class__._PATTERN_MUL_OR_DIV][0].name]
        next_node = graph.get_next_nodes(mul_or_div.outputs[0])[0]

        mul_or_div.inputs[0] = matmul.inputs[0]
        matmul.inputs[0] = mul_or_div.outputs[0]
        next_node.inputs[0] = matmul.outputs[0]

        return True


@knowledge_factory.KnowledgeFactory.register()
class KnowledgeFlashAttentionTik(KnowledgeBaseWithPostProcess):
    """
    Transpose
        |
      MatMul
        |        ==>        FlashAttentionTik
     Softmax
        |
      MatMul
    """
    _FA_TYPE = 'FlashAttentionTik'
    _PATTERN_TRANSPOSE = 'Transpose'
    _PATTERN_MATMUL = 'MatMul'
    _PATTERN_SOFTMAX = 'Softmax'
    _PATTERN_MATMUL_1 = 'MatMul_1'

    def __init__(self):
        super().__init__()
        self._register_apply_funcs(self._define_pattern(), [self._apply])

    def _define_pattern(self):
        return pattern_module.Pattern() \
            .add_node(
            self.__class__._PATTERN_TRANSPOSE,
            ['Transpose'],
            [pattern_utils.NextNodeCount(1), IsSecondInputOfMatMul()],
        ) \
            .add_node(self.__class__._PATTERN_MATMUL, ['MatMul'], [pattern_utils.NextNodeCount(1)]) \
            .add_node(self.__class__._PATTERN_SOFTMAX, ['Softmax'], [pattern_utils.NextNodeCount(1)]) \
            .add_node(self.__class__._PATTERN_MATMUL_1, ['MatMul'], [pattern_utils.NextNodeCount(1)]) \
            .add_edge(self.__class__._PATTERN_TRANSPOSE, self.__class__._PATTERN_MATMUL) \
            .add_edge(self.__class__._PATTERN_MATMUL, self.__class__._PATTERN_SOFTMAX) \
            .add_edge(self.__class__._PATTERN_SOFTMAX, self.__class__._PATTERN_MATMUL_1)

    def _apply(self, graph: base_graph.BaseGraph, match_result: matcher.MatchResult) -> bool:
        KnowledgeOptimizerFlashAttentionTik(graph, match_result).apply()
        return True


class KnowledgeOptimizerFlashAttentionTik(KnowledgeOptimizer):
    _knowledge_class = KnowledgeFlashAttentionTik

    def __init__(self, graph: base_graph.BaseGraph, match_result: matcher.MatchResult):
        super().__init__(graph, match_result)
        nodes = match_result.node_dicts[0]
        self._transpose = graph[nodes[self.__class__._knowledge_class._PATTERN_TRANSPOSE][0].name]
        self._matmul = graph[nodes[self.__class__._knowledge_class._PATTERN_MATMUL][0].name]
        self._softmax = graph[nodes[self.__class__._knowledge_class._PATTERN_SOFTMAX][0].name]
        self._matmul_1 = graph[nodes[self.__class__._knowledge_class._PATTERN_MATMUL_1][0].name]

    def apply(self) -> None:
        if self._transpose.attrs['perm'] != [0, 1, 3, 2]:
            self.__change_transpose_perm()
        self._add_fa()

    def __change_transpose_perm(self) -> None:
        new_transpose_name = self._transpose.name + '_before'
        new_transpose_perm = self._transpose.attrs['perm'].copy()
        new_transpose_perm[-2], new_transpose_perm[-1] = new_transpose_perm[-1], new_transpose_perm[-2]
        new_transpose = self._graph.add_node(
            new_transpose_name,
            'Transpose',
            inputs=[self._transpose.inputs[0]],
            outputs=[new_transpose_name + '_output'],
            attrs={'perm': new_transpose_perm},
        )
        self._transpose.inputs[0] = new_transpose.outputs[0]
        self._transpose.attrs['perm'] = [0, 1, 3, 2]

        output_shape = list(self._graph.get_value_info(self._transpose.outputs[0]).shape)
        output_shape[-2], output_shape[-1] = output_shape[-1], output_shape[-2]
        self._graph._value_infos.append(
            onnx_node.OnnxPlaceHolder(new_transpose.outputs[0], np.dtype('int64'), output_shape)
        )
        self._graph.update_map()

    def _add_fa(self) -> None:
        new_fa_name = self._softmax.name.replace('Softmax', self.__class__._knowledge_class._FA_TYPE)
        self._fa = self._graph.add_node(
            new_fa_name,
            self.__class__._knowledge_class._FA_TYPE,
            inputs=[self._matmul.inputs[0], self._transpose.inputs[0], self._matmul_1.inputs[1]],
            outputs=[new_fa_name + '_output'],
        )
        next_node = self._graph.get_next_nodes(self._matmul_1.outputs[0])[0]
        next_node.inputs[0] = self._fa.outputs[0]

        output_shape = list(self._graph.get_value_info(self._matmul.inputs[0]).shape)
        self._graph._value_infos.append(onnx_node.OnnxPlaceHolder(self._fa.outputs[0], np.dtype('int64'), output_shape))
        self._graph.update_map()


@knowledge_factory.KnowledgeFactory.register()
class KnowledgeFlashAttentionSoftmaxFp32(KnowledgeFlashAttentionTik):
    """
    Mul/Div Transpose
         \    /
         MatMul
           |
          Add        ==>        FlashAttentionSoftmaxFp32
           |
        Softmax
           |
         MatMul
    """
    _FA_TYPE = 'FlashAttentionSoftmaxFp32'
    _PATTERN_MUL_OR_DIV = 'Mul_or_Div'
    _PATTERN_ADD = 'Add'

    def __init__(self):
        super().__init__()
        self._register_apply_funcs(self._define_pattern(), [self._apply])

    def _define_pattern(self):
        pattern = super()._define_pattern()
        _remove_edge(pattern, self.__class__._PATTERN_MATMUL, self.__class__._PATTERN_SOFTMAX)
        pattern.add_node(self.__class__._PATTERN_MUL_OR_DIV, ['Mul', 'Div'], [pattern_utils.NextNodeCount(1)]) \
            .add_node(self.__class__._PATTERN_ADD, ['Add'], [pattern_utils.NextNodeCount(1)]) \
            .add_edge(self.__class__._PATTERN_MUL_OR_DIV, self.__class__._PATTERN_MATMUL) \
            .add_edge(self.__class__._PATTERN_MATMUL, self.__class__._PATTERN_ADD) \
            .add_edge(self.__class__._PATTERN_ADD, self.__class__._PATTERN_SOFTMAX)
        return pattern

    def _apply(self, graph: base_graph.BaseGraph, match_result: matcher.MatchResult) -> bool:
        KnowledgeOptimizerFlashAttentionSoftmaxFp32(graph, match_result).apply()
        return True


class KnowledgeOptimizerFlashAttentionSoftmaxFp32(KnowledgeOptimizerFlashAttentionTik):
    _knowledge_class = KnowledgeFlashAttentionSoftmaxFp32

    def __init__(self, graph: base_graph.BaseGraph, match_result: matcher.MatchResult):
        super().__init__(graph, match_result)
        nodes = match_result.node_dicts[0]
        self._mul_or_div = graph[nodes[self.__class__._knowledge_class._PATTERN_MUL_OR_DIV][0].name]
        self._add = graph[nodes[self.__class__._knowledge_class._PATTERN_ADD][0].name]
        self.__name_prefix = '/'.join(self._softmax.name.split('/')[:-1]) + '/'

    def apply(self) -> None:
        op_type = self._mul_or_div.op_type
        value = self._graph[self._mul_or_div.inputs[1]].value
        matched = (op_type == 'Mul' and value == 0.125) or (op_type == 'Div' and value == 8)
        if not matched:
            return

        super().apply()
        self._graph.remove_unused_nodes()

        q_shape = self._graph.get_value_info(self._matmul.inputs[0]).shape
        if len(q_shape) == 4:
            # FlashAttentionTik supports both 3-dim and 4-dim inputs, while FlashAttentionSoftmaxFp32 supports only
            # 3-dim inputs.
            self.__add_reshapes()

    def _add_fa(self) -> None:
        super()._add_fa()
        q_seq_len = self.__add_seq_len_initializer(self._matmul.inputs[0], 'q_seq_len')
        kv_seq_len = self.__add_seq_len_initializer(self._transpose.inputs[0], 'kv_seq_len')
        unsqueeze_mask = self.__add_unsqueeze(self._add.inputs[1])

        self._fa.inputs[0] = self._mul_or_div.inputs[0]
        self._fa.inputs.extend([q_seq_len.name, kv_seq_len.name, unsqueeze_mask.outputs[0]])

    def __add_seq_len_initializer(self, data_name: str, initializer_name: str) -> onnx_node.OnnxInitializer:
        seq_len = self._graph.get_value_info(data_name).shape[-2]
        return self._graph.add_initializer(
            self.__name_prefix + initializer_name,
            np.array([seq_len], dtype=np.int32),
        )

    def __add_reshapes(self) -> None:
        q_shape = self._graph.get_value_info(self._matmul.inputs[0]).shape
        first_dim = q_shape[0]
        self.__add_reshape('combine_dims', 'Reshape_q', self._mul_or_div.inputs[0])
        self.__add_reshape('combine_dims', 'Reshape_k', self._transpose.inputs[0])
        self.__add_reshape('combine_dims', 'Reshape_v', self._matmul_1.inputs[1])

        mask = self._add.inputs[1]
        unsqueeze_mask = self._graph.get_prev_node(mask)
        self.__add_reshape('combine_dims', 'Reshape_mask', unsqueeze_mask.inputs[0])

        self._graph.get_value_info(self._fa.outputs[0]).shape \
            = self._graph[self.__name_prefix + 'Reshape_q_shape'].value.tolist()
        self.__add_reshape('split_dims', 'Reshape_fa_output', self._fa.outputs[0], first_dim)

    def __add_reshape(self, type_: str, node_name: str, input_name: str, first_dim: int = None) -> base_node.BaseNode:
        node_name = self.__name_prefix + node_name

        shape = list(self._graph.get_value_info(input_name).shape)
        if type_ == 'combine_dims':
            shape[:2] = [shape[0] * shape[1]]
        elif type_ == 'split_dims':
            shape[:1] = [first_dim, shape[0] // first_dim]
        initializer = self._graph.add_initializer(node_name + '_shape', np.array(shape))

        next_node = self._graph.get_next_nodes(input_name)[0]
        reshape = self._graph.add_node(
            node_name,
            'Reshape',
            inputs=[input_name, initializer.name],
            outputs=[node_name + '_output'],
        )
        _replace_input(next_node, input_name, reshape.outputs[0])

        return reshape

    def __add_unsqueeze(self, mask: str) -> base_node.BaseNode:
        origin_shape = self._graph.get_value_info(mask).shape
        target_shape = self._graph.get_value_info(self._add.inputs[0]).shape
        prev_node = self._graph.get_prev_node(mask)

        if origin_shape != target_shape:
            repeats_value = [target_shape[i] // origin_shape[i] for i in range(len(target_shape))]
            repeats = self._graph.add_initializer(self.__name_prefix + 'mask_repeats', np.array(repeats_value))
            tile_name = self.__name_prefix + 'Tile_mask'
            tile = self._graph.add_node(
                tile_name,
                'Tile',
                inputs=[prev_node.outputs[0], repeats.name],
                outputs=[tile_name + '_output'],
            )
            output_shape = list(target_shape)
            self._graph._value_infos.append(onnx_node.OnnxPlaceHolder(tile.outputs[0], np.dtype('int64'), output_shape))
            self._graph.update_map()
            prev_node = tile

        unsqueeze_name = self.__name_prefix + 'Unsqueeze_mask'
        unsqueeze = self._graph.add_node(
            unsqueeze_name,
            'Unsqueeze',
            inputs=[prev_node.outputs[0]],
            outputs=[unsqueeze_name + '_output'],
            attrs={'axes': [1]},
        )
        self._add.inputs[1] = unsqueeze.outputs[0]

        return unsqueeze


@knowledge_factory.KnowledgeFactory.register()
class KnowledgeMultiTile(KnowledgeBaseWithPostProcess):
    """
                  Mul
                  /|  \
                 / |    \
                /  |      \
               /   |        \                        Mul
              /    |          \                       |
             /     |            \                    Tile
            /      |              \        ==>        |
           /       |                \              Reshape
       Tile       Tile            Tile                |
        |          |               |              Unsqueeze
     Reshape    Reshape   ...   Reshape
        |          |               |
    Unsqueeze  Unsqueeze       Unsqueeze
    """
    _PATTERN_MUL = 'Mul'
    _PATTERN_TILE = 'Tile'
    _PATTERN_RESHAPE = 'Reshape'
    _PATTERN_UNSQUEEZE = 'Unsqueeze'
    _LAYER_NUM = 12

    def __init__(self):
        super().__init__()
        self._register_apply_funcs(self._define_pattern(), [self._apply])

    def _define_pattern(self):
        return pattern_module.Pattern().add_node(
            self.__class__._PATTERN_MUL,
            ['Mul'],
            [pattern_utils.NextNodeCount(self.__class__._LAYER_NUM)],
        )

    def _apply(self, graph: base_graph.BaseGraph, match_result: matcher.MatchResult) -> bool:
        nodes = match_result.node_dicts[0]
        mul = graph[nodes[self.__class__._PATTERN_MUL][0].name]
        tile_0 = graph.get_next_nodes(mul.outputs[0])[0]
        reshape_0 = graph.get_next_nodes(tile_0.outputs[0])[0]
        unsqueeze_0 = graph.get_next_nodes(reshape_0.outputs[0])[0]

        for i in range(1, self.__class__._LAYER_NUM):
            tile_i = graph.get_next_nodes(mul.outputs[0])[i]
            reshape_i = graph.get_next_nodes(tile_i.outputs[0])[0]
            unsqueeze_i = graph.get_next_nodes(reshape_i.outputs[0])[0]
            next_node = graph.get_next_nodes(unsqueeze_i.outputs[0])[0]
            _replace_input(next_node, unsqueeze_i.outputs[0], unsqueeze_0.outputs[0])

        return True


def _modify_visual_encoder(model_path, new_model_path):
    graph = auto_optimizer.OnnxGraph.parse(model_path)

    optimizer.GraphOptimizer.optimize(graph, KnowledgeMoveMulOrDivBeforeMatmul())
    optimizer.GraphOptimizer.optimize(graph, KnowledgeFlashAttentionTik())
    optimizer.GraphOptimizer.optimize(graph, knowledges.KnowledgeGatherToSplit())

    # Reshape data to 2-dim to accelerate the MatMul nodes and reduce the TransData nodes.
    # 1. A MatMul node will be faster if its first input is 2-dim.
    # 2. A LayerNorm node can process its input in NZ format only when the last two dims are 16-aligned. Otherwise,
    #    TransData nodes need to be inserted to convert the input to ND format.
    concat = graph.get_nodes('Concat')[0]
    add = graph.get_next_nodes(concat.outputs[0])[0]
    batch_size, seq_len, head_dim = graph.get_value_info(add.inputs[0]).shape
    two_dim_shape = graph.add_initializer('two_dim_shape', np.array([batch_size * seq_len, head_dim]))
    three_dim_shape = graph.add_initializer('three_dim_shape', np.array([batch_size, seq_len, head_dim]))
    # Add a Reshape node after the last LayerNorm node.
    reshape_before = graph.add_node(
        'Reshape_before',
        'Reshape',
        inputs=[add.outputs[0], two_dim_shape.name],
        outputs=['Reshape_before_output'],
    )
    for next_node in graph.get_next_nodes(add.outputs[0]):
        _replace_input(next_node, add.outputs[0], reshape_before.outputs[0])
    # Change the shape of the Reshape node after every Attention structure.
    graph.update_map()  # Make sure the newly added Reshape nodes has correct outputs.
    for reshape in graph.get_nodes('Reshape'):
        if graph.get_next_nodes(reshape.outputs[0])[0].op_type == 'MatMul':
            reshape.inputs[1] = two_dim_shape.name
    # Add a Reshape node after the last LayerNorm node.
    output = graph.outputs[0]
    add = graph.get_prev_node(output.name)
    add.outputs[0] = 'Add_output'
    graph.add_node(
        'Reshape_after',
        'Reshape',
        inputs=[add.outputs[0], three_dim_shape.name],
        outputs=[output.name],
    )

    # A Conv node's output is in NC1HWC0 format, and it will be faster if the W-dim of its output is 16-aligned. So pad
    # data to make it 16-aligned.
    conv = graph.get_nodes('Conv')[0]
    old_input_w = graph[conv.inputs[0]].shape[-1]
    old_output_w = graph.get_value_info(conv.outputs[0]).shape[-1]
    block_size = 16
    new_output_w = (old_output_w + block_size - 1) // block_size * block_size
    stride_w = conv.attrs['strides'][1]
    kernel_w = graph[conv.inputs[1]].value.shape[-1]
    pad_w = conv.attrs['pads'][2] + conv.attrs['pads'][3]
    # From $W_{out} = \frac{W + p_w - k_w}{s_w} + 1$, we get $W = (W_{out} - 1)s_w + k_w - p_w$.
    new_input_w = (new_output_w - 1) * stride_w + kernel_w - pad_w
    reshape = graph.get_next_nodes(conv.outputs[0])[0]
    input_ = conv.inputs[0]
    pad_before_initializer = graph.add_initializer(
        'pad_before_initializer',
        np.array([0, 0, 0, 0, 0, 0, 0, new_input_w - old_input_w]),
    )
    pad_before = graph.add_node(
        'Pad_before',
        'Pad',
        inputs=[input_, pad_before_initializer.name],
        outputs=['Pad_before_output'],
    )
    conv.inputs[0] = pad_before.outputs[0]
    split_after = graph.add_node(
        'Split_after',
        'Split',
        inputs=[conv.outputs[0]],
        outputs=['Split_after_output_0', 'Split_after_output_1'],
        attrs={'axis': 3, 'split': [old_input_w // block_size, (new_input_w - old_input_w) // block_size]},
    )
    reshape.inputs[0] = split_after.outputs[0]

    graph.save(new_model_path)


def _modify_text_encoder(model_path, new_model_path):
    graph = auto_optimizer.OnnxGraph.parse(model_path)

    # Remove the unused inputs.
    next_nodes = graph.get_next_nodes('image_atts')
    while len(next_nodes) == 1:
        next_nodes = graph.get_next_nodes(next_nodes[0].outputs[0])
    # Create a tuple to prevent the list from changing during node deletion.
    for node in tuple(next_nodes):
        graph.remove(node.name)
    graph.remove_unused_nodes()

    optimizer.GraphOptimizer.optimize(graph, KnowledgeMoveMulOrDivBeforeMatmul())
    optimizer.GraphOptimizer.optimize(graph, KnowledgeFlashAttentionTik())
    optimizer.GraphOptimizer.optimize(graph, KnowledgeFlashAttentionSoftmaxFp32())
    optimizer.GraphOptimizer.optimize(graph, KnowledgeMultiTile())

    # Reshape data to 2-dim to accelerate the MatMul nodes and reduce the TransData nodes.
    # 1. A MatMul node will be faster if its first input is 2-dim.
    # 2. A LayerNorm node can process its input in NZ format only when the last two dims are 16-aligned. Otherwise,
    #    TransData nodes need to be inserted to convert the input to ND format.
    node = graph.get_next_nodes('input_ids')[0]
    while node.op_type != 'Add':
        node = graph.get_next_nodes(node.outputs[0])[0]
    add = node
    batch_size, seq_len, head_dim = graph.get_value_info(add.inputs[0]).shape
    two_dim_shape = graph.add_initializer('two_dim_shape', np.array([batch_size * seq_len, head_dim]))
    three_dim_shape = graph.add_initializer('three_dim_shape', np.array([batch_size, seq_len, head_dim]))
    # Add a Reshape node before the first LayerNorm node.
    reshape_before = graph.add_node(
        'Reshape_before',
        'Reshape',
        inputs=[add.outputs[0], two_dim_shape.name],
        outputs=['Reshape_before_output'],
    )
    for next_node in graph.get_next_nodes(add.outputs[0]):
        _replace_input(next_node, add.outputs[0], reshape_before.outputs[0])
    # Change the shape of the Reshape node after every Attention structure.
    for fa in graph.get_nodes('FlashAttentionTik'):
        transpose = graph.get_next_nodes(fa.outputs[0])[0]
        reshape = graph.get_next_nodes(transpose.outputs[0])[0]
        reshape.inputs[1] = two_dim_shape.name
    for fa in graph.get_nodes('FlashAttentionSoftmaxFp32'):
        reshape_fa_output = graph.get_next_nodes(fa.outputs[0])[0]
        transpose = graph.get_next_nodes(reshape_fa_output.outputs[0])[0]
        reshape = graph.get_next_nodes(transpose.outputs[0])[0]
        reshape.inputs[1] = two_dim_shape.name
    # Add a Reshape node after the last LayerNorm node.
    output = graph.outputs[0]
    node = graph.get_prev_node(output.name)
    node.outputs[0] = 'Add_output'
    reshape_after = graph.add_node(
        'Reshape_after',
        'Reshape',
        inputs=[node.outputs[0], three_dim_shape.name],
        outputs=[output.name],
    )

    # Why don't reshape data to 2-dim to accelerate the MatMul nodes after the input image_embeds?
    # If you do this, it will cause the MTE2 time of the Transpose nodes after the MatMul nodes to taker longer, and the
    # overall time will also take longer.

    graph.save(new_model_path)


def _modify_text_decoder_1(model_path, new_model_path):
    graph = auto_optimizer.OnnxGraph.parse(model_path)

    # Why don't apply KnowledgeFlashAttentionTik and KnowledgeFlashAttentionSoftmaxFp32?
    # q_seq_len (1) is too small, and the performance will deteriorate after application.

    # Reshape data to 2-dim to accelerate the MatMul nodes. A MatMul node will be faster if its first input is 2-dim.
    question_states = graph['question_states']
    batch_size, seq_len, head_dim = graph[question_states.name].shape
    two_dim_shape = graph.add_initializer('two_dim_shape', np.array([batch_size * seq_len, head_dim]))
    reshape_before = graph.add_node(
        'Reshape_before',
        'Reshape',
        inputs=[question_states.name, two_dim_shape.name],
        outputs=['Reshape_before_output'],
    )
    for next_node in graph.get_next_nodes(question_states.name):
        _replace_input(next_node, question_states.name, reshape_before.outputs[0])

    graph.save(new_model_path)


def _modify_text_decoder_2(model_path, new_model_path):
    graph = auto_optimizer.OnnxGraph.parse(model_path)

    optimizer.GraphOptimizer.optimize(graph, KnowledgeMoveMulOrDivBeforeMatmul())
    optimizer.GraphOptimizer.optimize(graph, KnowledgeFlashAttentionSoftmaxFp32())
    optimizer.GraphOptimizer.optimize(graph, KnowledgeMultiTile())

    # The SoftmaxCrossEntropyLoss operator on Ascend hasn't supported the ignore_index attribute yet. So add additional
    # nodes to achieve the same functionality.
    node = graph.get_nodes('SoftmaxCrossEntropyLoss')[0]
    graph.add_initializer('ignore_index', np.array(node.attrs['ignore_index']))
    graph.add_initializer('zero', np.array(0).astype(np.float32))
    graph.add_initializer('one', np.array(1).astype(np.float32))
    graph.add_node(
        'Equal_Label',
        'Equal',
        inputs=[node.inputs[1], 'ignore_index'],
        outputs=['Equal_Label_Output']
    )
    graph.add_node(
        'Where_pad',
        'Where',
        inputs=['Equal_Label_Output', 'zero', 'one'],
        outputs=['Where_Pad_Output']
    )
    mul = graph.add_node('Mul_Mask', 'Mul')
    graph.insert_node(node.name, mul, mode='after')
    mul.inputs.append('Where_Pad_Output')

    # Remove the unused output.
    graph.remove(graph.outputs[1].name)

    graph.save(new_model_path)


def main(model_dir):
    visual_encoder_sim = os.path.join(model_dir, 'visual_encoder_sim.onnx')
    visual_encoder_md = os.path.join(model_dir, 'visual_encoder_md.onnx')
    _modify_visual_encoder(visual_encoder_sim, visual_encoder_md)

    text_encoder_sim = os.path.join(model_dir, 'text_encoder_sim.onnx')
    text_encoder_md = os.path.join(model_dir, 'text_encoder_md.onnx')
    _modify_text_encoder(text_encoder_sim, text_encoder_md)

    text_decoder_1_sim = os.path.join(model_dir, 'text_decoder_1_sim.onnx')
    text_decoder_1_md = os.path.join(model_dir, 'text_decoder_1_md.onnx')
    _modify_text_decoder_1(text_decoder_1_sim, text_decoder_1_md)

    text_decoder_2_sim = os.path.join(model_dir, 'text_decoder_2_sim.onnx')
    text_decoder_2_md = os.path.join(model_dir, 'text_decoder_2_md.onnx')
    _modify_text_decoder_2(text_decoder_2_sim, text_decoder_2_md)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='blip_models',
        help='Path of ONNX models.',
    )
    args = parser.parse_args()
    main(args.model_dir)
