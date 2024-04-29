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

import os
import sys
import argparse

import yaml
import torch
import numpy as np

from models.blip_vqa import blip_vqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str, 
        default='./configs/vqa.yaml',
        help='Path of config file.',
    )
    parser.add_argument(
        '--pth_path', 
        type=str, 
        default='./model_base_vqa_capfilt_large.pth',
        help='Path or name of the pre-trained model.',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='blip_models',
        help='Path of directory to save ONNX models.',
    )
    return parser.parse_args()


def load_model(pth, config):
    if not os.path.exists(pth):
        print("The pth does not exist. Download pth...")
        pth = config['pretrained']

    model = blip_vqa(
        pretrained=pth,
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
    )
    model.eval()

    return model


def export_visual_encoder(model, image_size, output_dir):
    dummy_input = torch.rand([1, 3, image_size, image_size])
    print('Exporting the visual encoder...')
    torch.onnx.export(
        model.visual_encoder,
        dummy_input,
        os.path.join(output_dir, 'visual_encoder.onnx'),
        input_names=['image'],
        output_names=['image_embeds'],
        dynamic_axes={'image': {0: 'bs'}},
        verbose=False,
        opset_version=11,
    )
    print('Done.')


def export_text_encoder(model, output_dir):
    dummy_input = (
        torch.ones([1, 35], dtype=torch.int64),
        torch.ones([1, 35], dtype=torch.int64),
        {
            'encoder_hidden_states': torch.rand([1, 901, 768]),
            'encoder_attention_mask': torch.ones([1, 901], dtype=torch.int64),
            'return_dict': True,
        }
    )
    print('Exporting the text encoder...')
    torch.onnx.export(
        model.text_encoder,
        dummy_input,
        os.path.join(output_dir, 'text_encoder.onnx'),
        input_names=['input_ids', 'attention_mask', 
                    'image_embeds', 'image_atts'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'bs', 1: 'seq_len'},
            'attention_mask': {0: 'bs', 1: 'seq_len'}, 
            'image_embeds': {0: 'bs'}, 
            'image_atts': {0: 'bs'},
        },
        verbose=False,
        opset_version=11,
    )
    print('Done.')


def export_text_decoder(model, k_test, output_dir):
    dummy_input_1 = (
        torch.ones([1, 1], dtype=torch.int64),
        {
            'encoder_hidden_states': torch.rand([1, 35, 768]),
            'encoder_attention_mask': torch.ones([1, 35], dtype=torch.int64),
            'return_logits': True,
        }
    )
    print('Exporting the text decoder...')
    torch.onnx.export(
        model.text_decoder,
        dummy_input_1,
        os.path.join(output_dir, 'text_decoder_1.onnx'),
        input_names=['start_ids', 'question_states', 'question_atts'],
        output_names=['start_output'],
        dynamic_axes={
            'start_ids': {0: 'bs'},
            'question_states': {0: 'bs', 1: 'seq_len'}, 
            'question_atts': {0: 'bs', 1: 'seq_len'}, 
        },
        verbose=False,
        opset_version=11,
    )

    dummy_input_2 = (
        torch.ones([k_test, 8], dtype=torch.int64),
        torch.ones([k_test, 8], dtype=torch.int64),
        {
            'encoder_hidden_states': torch.rand([k_test, 35, 768]),
            'encoder_attention_mask': torch.ones([k_test, 35], dtype=torch.int64),
            'labels': torch.ones([k_test, 8], dtype=torch.int64),
            'return_dict': True,
            'reduction': 'none',
        }
    )
    torch.onnx.export(
        model.text_decoder,
        dummy_input_2,
        os.path.join(output_dir, 'text_decoder_2.onnx'),
        input_names=['input_ids', 'input_atts', 'question_states', 
                    'question_atts', 'target_ids'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'k_test'},
            'input_atts': {0: 'k_test'}, 
            'question_states': {0: 'k_test', 1: 'seq_len'}, 
            'question_atts': {0: 'k_test', 1: 'seq_len'},
            'target_ids': {0: 'k_test'}, 
        },
        verbose=False,
        opset_version=12,
    )
    print('Done.')



def main(args):
    config = yaml.safe_load(open(args.config, 'r'))
    model = load_model(args.pth_path, config)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    export_visual_encoder(model, config['image_size'], args.output_dir)
    export_text_encoder(model, args.output_dir)
    export_text_decoder(model, config['k_test'], args.output_dir)


if __name__ == '__main__':
    main(parse_args())
