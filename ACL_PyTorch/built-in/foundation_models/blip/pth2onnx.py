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

import os
import sys
import argparse

import yaml
import torch
import numpy as np

from models.blip import blip_decoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str, 
        default='./configs/caption_coco.yaml',
        help='Path of config file.',
        )
    parser.add_argument(
        '--pth_path', 
        type=str, 
        default='./model_base_caption_capfilt_large.pth',
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
        pth = config['pretrained']

    model = blip_decoder(
        pretrained=pth,
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
        prompt=config['prompt']
    )
    model.eval()

    return model


def export_onnx(model, image_size, output_dir):
    dummy_input_encoder = torch.rand([1, 3, image_size, image_size])
    print('Exporting the visual encoder...')
    torch.onnx.export(
        model.visual_encoder,
        dummy_input_encoder,
        os.path.join(output_dir, 'visual_encoder.onnx'),
        input_names=['image'],
        output_names=['image_embeds'],
        dynamic_axes={'image': {0: '-1'}},
        verbose=False,
        opset_version=11
    )
    print('Done.')

    dummy_input_decoder = (
        torch.randint(100, [3, 4]),
        torch.ones([3, 4], dtype=torch.int64),
        {
            'encoder_hidden_states': torch.rand([3, 577, 768]),
            'encoder_attention_mask': torch.ones([3, 577], dtype=torch.int64)
        }
    )
    print('Exporting the text decoder...')
    torch.onnx.export(
        model.text_decoder,
        dummy_input_decoder,
        os.path.join(output_dir, 'text_decoder.onnx'),
        input_names=['input_ids', 'attention_mask', 
                    'encoder_hidden_states', 'encoder_attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'bs', 1: 'seq_len'},
            'attention_mask': {0: 'bs', 1: 'seq_len'}, 
            'encoder_hidden_states': {0: 'bs'}, 
            'encoder_attention_mask': {0: 'bs'},
        },
        verbose=False,
        opset_version=11
    )
    print('Done.')


def main(args):
    config = yaml.safe_load(open(args.config, 'r'))
    model = load_model(args.pth_path, config)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    export_onnx(model, config['image_size'], args.output_dir)


if __name__ == '__main__':
    main(parse_args())
