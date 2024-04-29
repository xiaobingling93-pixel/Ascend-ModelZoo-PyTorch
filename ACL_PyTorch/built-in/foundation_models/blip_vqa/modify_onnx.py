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

import numpy as np
from auto_optimizer import OnnxGraph


def replace_fa(model, matmul, mul, softmax, matmul_1, transpose=None):
    """
    pattern1:
        |           \        /                                   |         |          |
        |            \      /                                    Mul       |         /
         \            MatMul                                       \       |        /
          \             |                                           \      |       /
           \           Mul                =======>                   FlashAttentionTik
            \           |                                                  |
             \       SoftMax                                               |
              \        /                                                   |
                MatMul                                                     |
                  |
    pattern2:
        |          \          /
        |           \     transpose                              |         |          |
        |            \      /                                    Div       |         /
         \            MatMul                                       \       |        /
          \             |                                           \      |       /
           \           Div                =======>                FlashAttentionTik
            \           |                                                  |
             \       SoftMax                                               |
              \        /                                                   |
                MatMul                                                     |
                  |
    """
    fa = 'FlashAttentionTik'
    # move mul before matmul
    softmax.inputs[0] = matmul.outputs[0]
    mul.inputs[0] = matmul.inputs[0]
    matmul.inputs[0] = mul.outputs[0]

    # add flashattention
    new_node = model.add_node(
        softmax.name.replace('Softmax', fa),
        fa,
        inputs=[matmul.inputs[0], matmul.inputs[1], matmul_1.inputs[1]],
        outputs=[matmul_1.outputs[0]]
    )

    # remove old nodes
    model.remove(matmul.name, {})
    model.remove(matmul_1.name, {})
    model.remove(softmax.name, {})

    if transpose:
        new_node.inputs[1] = transpose.inputs[0]
        model.remove(transpose.name, {})
    
    model.update_map()
    return new_node


def modify_visual_encoder(model_path, new_model_path):
    model = OnnxGraph.parse(model_path)

    for node in model.get_nodes('Softmax'):
        # use FA
        mul = model.get_prev_node(node.inputs[0])
        matmul = model.get_prev_node(mul.inputs[0])
        transpose = model.get_prev_node(matmul.inputs[1])
        matmul_1 = model.get_next_nodes(node.outputs[0])[0]

        new_node = replace_fa(model, matmul, mul, node, matmul_1, transpose)

        # use split
        input_name = model.get_prev_node(mul.inputs[0]).inputs[0]
        gathers = [n.name for n in model.get_next_nodes(input_name)]
        for gather in gathers:
            model.remove(gather, {})

        name_split = node.name.replace('Softmax', 'Split')
        model.add_node(
            name_split,
            'Split',
            attrs={'axis': 0, 'split': [1, 1, 1]},
            inputs=[input_name],
            outputs=[
                name_split + '_Output_0',
                name_split + '_Output_1',
                name_split + '_Output_2',
            ]
        )
        
        name_sq = node.name.replace('Softmax', 'Squeeze_')
        for i in range(3):
            name = name_sq + str(i)
            model.add_node(
                name,
                'Squeeze',
                attrs={'axes': 0},
                inputs=[f'{name_split}_Output_{i}'],
                outputs=[f'{name_sq}_{i}_Output_0'],
            )
        mul.inputs[0] = name_sq + '_0_Output_0'
        new_node.inputs[1] = name_sq + '_1_Output_0'
        new_node.inputs[2] = name_sq + '_2_Output_0'

    model.update_map()
    model.save(new_model_path)


def modify_text_encoder(model_path, new_model_path, batch_size):
    model = OnnxGraph.parse(model_path)

    # remove unused input
    next_nodes = model.get_next_nodes('image_atts')
    del_nodes = []
    while len(next_nodes) == 1:
        del_nodes.extend(next_nodes)
        next_nodes = model.get_next_nodes(next_nodes[0].outputs[0])

    for node in del_nodes:
        model.remove(node.name, {})
    for node in next_nodes:
        model.remove(node.name)
    
    model.update_map()
    model.remove_unused_nodes()

    # add reshape to reduce transdata
    reshapes = [n.name for n in model.get_nodes('Reshape')]
    model.add_initializer(
        'original_shape', 
        np.array([batch_size, -1, 12, 64]).astype(np.int64),
    )
    model.add_initializer(
        'output_shape', 
        np.array([batch_size, -1, 768]).astype(np.int64),
    )
    model.add_initializer(
        'shape_constant', 
        np.array([-1, 768]).astype(np.int64),
    )
    # add reshape before the first layernorm
    reshape_1 = model.add_node('Reshape_before', 'Reshape')
    node = model.get_next_nodes('input_ids')[0]
    while node.op_type != 'Add':
        node = model.get_next_nodes(node.outputs[0])[0]
    model.insert_node(node.name, reshape_1)
    reshape_1.inputs.append('shape_constant')
    # add reshape after the last layernorm
    reshape_2 = model.add_node('Reshape_after', 'Reshape')
    node = model.get_prev_node(model.outputs[0].name)
    model.insert_node(node.name, reshape_2)
    reshape_2.inputs.append('output_shape')

    for node in model.get_nodes('Softmax'):
        div = model.get_prev_node(node.inputs[0])
        matmul = model.get_prev_node(div.inputs[0])
        matmul_1 = model.get_next_nodes(node.outputs[0])[0]
        # change shape of reshape node after attention
        next_node = model.get_next_nodes(matmul_1.outputs[0])[0]
        next_nodes = model.get_next_nodes(next_node.outputs[0])
        for next_node in next_nodes:
            if next_node.op_type == 'Reshape':
                break
        next_node.inputs[1] = 'shape_constant'
        reshapes.remove(next_node.name)
        # use fa
        if div.op_type != 'Div':
            continue
        transpose = model.get_prev_node(matmul.inputs[1])
        transpose.attrs['perm'] = [0, 2, 1, 3]
        replace_fa(model, matmul, div, node, matmul_1)

    for reshape in reshapes:
        model[reshape].inputs[1] = 'original_shape'
    model.update_map()
    model.remove_unused_nodes()

    model.save(new_model_path)
    

def modify_decoder_2(model_path, new_model_path):
    model = OnnxGraph.parse(model_path)
    # add ignore_indx function
    node = model.get_nodes('SoftmaxCrossEntropyLoss')[0]
    model.add_initializer('ignore_index', np.array(node.attrs['ignore_index']))
    model.add_initializer('zero', np.array(0).astype(np.float32))
    model.add_initializer('one', np.array(1).astype(np.float32))
    model.add_node(
        'Equal_Label', 
        'Equal', 
        inputs=[node.inputs[1], 'ignore_index'], 
        outputs=['Equal_Label_Output']
    )
    model.add_node(
        'Where_pad', 
        'Where', 
        inputs=['Equal_Label_Output', 'zero', 'one'], 
        outputs=['Where_Pad_Output']
    )
    mul = model.add_node('Mul_Mask', 'Mul')
    model.insert_node(node.name, mul, mode='after')
    mul.inputs.append('Where_Pad_Output')

    # remove unused output
    model.remove(model.outputs[1].name)

    model.save(new_model_path)

    
def main(model_dir, batch_size):
    visual_encoder = os.path.join(model_dir, 'visual_encoder.onnx')
    visual_encoder_md = os.path.join(model_dir, 'visual_encoder_md.onnx')
    modify_visual_encoder(visual_encoder, visual_encoder_md)

    text_encoder = os.path.join(model_dir, 'text_encoder.onnx')
    text_encoder_md = os.path.join(model_dir, 'text_encoder_md.onnx')
    modify_text_encoder(text_encoder, text_encoder_md, batch_size)

    text_decoder_2 = os.path.join(model_dir, 'text_decoder_2.onnx')
    text_decoder_2_md = os.path.join(model_dir, 'text_decoder_2_md.onnx')
    modify_decoder_2(text_decoder_2, text_decoder_2_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', 
        type=str, 
        default='blip_models',
        help='Path of ONNX models.',
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Batch size of data loader.',
    )
    args = parser.parse_args()
    main(args.model_dir, args.batch_size)
