# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd
# This software is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import torch
from torch import nn
import torch.nn.functional as F

from models.med import BertConfig, BertModel
from models.blip import create_vit, init_tokenizer, load_checkpoint


class BLIP_ITM_TEXT(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(
            config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, text_ids, text_atten_mask):
        text_output = self.text_encoder(text_ids, attention_mask=text_atten_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(
            text_output.last_hidden_state[:, 0, :]), dim=-1)
        return text_feat


def blip_itm_text(pretrained='', **kwargs):
    model = BLIP_ITM_TEXT(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert(len(msg.missing_keys) == 0)

    return model, model.tokenizer


class BLIP_ITM_IMAGE(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(
            config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, image):
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(
            self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat


def blip_itm_image(pretrained='', **kwargs):
    model = BLIP_ITM_IMAGE(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert(len(msg.missing_keys) == 0)

    return model, model.tokenizer


class BLIP_ITM_IMAGE_FEAT(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(
            vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(
            config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, image):
        image_feat = self.visual_encoder(image)
        return image_feat


def blip_itm_image_feat(pretrained='', **kwargs):
    model = BLIP_ITM_IMAGE_FEAT(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert(len(msg.missing_keys) == 0)

    return model, model.tokenizer


def pth2onnx_text(input_file, output_file, device='cpu'):

    model, tokenizer = blip_itm_text(
        pretrained=input_file, image_size=384, vit='base')

    model.eval()
    model.to(device)

    input_caption = ['Here is a apple']

    text = tokenizer(input_caption, padding='max_length',
                     truncation=True, max_length=35, return_tensors="pt")
    text_ids = text.input_ids.to(torch.int32)
    text_atten_mask = text.attention_mask.to(torch.int32)

    dummy_input = (text_ids, text_atten_mask)

    # 输入节点名
    input_names = ["text_ids", "text_atten_mask"]
    # 输出节点名
    output_names = ["output"]
    dynamic_axes = {'text_ids': {0: '-1'},
                    'text_atten_mask': {0: '-1'},
                    'output': {0: '-1'}
                    }

    output = model(*dummy_input)
    print(output.shape)
    # verbose=True，支持打印onnx节点和对应的PyTorch代码行
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True, dynamic_axes=dynamic_axes)

    print("Exporting .pth (text) model to onnx model has been successful!")


def pth2onnx_image(input_file, output_file, device='cpu'):

    model, tokenizer = blip_itm_image(
        pretrained=input_file, image_size=384, vit='base')

    model.eval()
    model.to(device)

    input_image = torch.randn(1, 3, 384, 384)
    dummy_input = input_image

    # 输入节点名
    input_names = ["input_image"]
    # 输出节点名
    output_names = ["output"]
    dynamic_axes = {'input_image': {0: '-1'},
                    'output': {0: '-1'}
                    }

    output = model(dummy_input)
    print(output.shape)
    # verbose=True，支持打印onnx节点和对应的PyTorch代码行
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True, dynamic_axes=dynamic_axes)

    print("Exporting .pth (image) model to onnx model has been successful!")


def pth2onnx_image_feat(input_file, output_file, device='cpu'):

    model, tokenizer = blip_itm_image_feat(
        pretrained=input_file, image_size=384, vit='base')

    model.eval()
    model.to(device)

    input_image = torch.randn(1, 3, 384, 384)
    dummy_input = input_image

    # 输入节点名
    input_names = ["input_image"]
    # 输出节点名
    output_names = ["output"]
    dynamic_axes = {'input_image': {0: '-1'},
                    'output': {0: '-1'}
                    }

    output = model(dummy_input)
    print(output.shape)
    # verbose=True，支持打印onnx节点和对应的PyTorch代码行
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names,
                      output_names=output_names, opset_version=11, verbose=True, dynamic_axes=dynamic_axes)

    print("Exporting .pth (image_feat) model to onnx model has been successful!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pth_path', default='./model_base_retrieval_coco.pth')
    args = parser.parse_args()

    pth2onnx_text(args.pth_path, "BLIP_text.onnx")
    pth2onnx_image(args.pth_path, "BLIP_image.onnx")
    pth2onnx_image_feat(args.pth_path, "BLIP_image_feat.onnx")
