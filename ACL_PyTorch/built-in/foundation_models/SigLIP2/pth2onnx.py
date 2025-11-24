# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import argparse
import glob
import torch
import torch.onnx
from transformers import AutoModel

os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)


class WrappedSigLIP2TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        return self.get_text_features(input_ids)

    def get_text_features(self, input_ids):
        outputs = self.model.get_text_features(input_ids=input_ids)
        return outputs


class WrappedSigLIP2VisionEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values):
        return self.get_image_features(pixel_values)

    def get_image_features(self, pixel_values):
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pytorch_ckpt_path',
        type=str,
        default='siglip2-so400m-patch14-384',
        help='Path or name of the pre-trained model weight dir.',
    )
    parser.add_argument(
        '--save_onnx_path',
        type=str,
        default='siglip2_onnx_models',
        help='Path of directory to save ONNX models.',
    )
    parser.add_argument(
        "--convert_text",
        action="store_true",
        help="Whether to convert the model's text feature extractor into ONNX."
    )
    parser.add_argument(
        "--convert_vision",
        action="store_true",
        help="Whether to convert the model's vision feature extractor into ONNX."
    )
    return parser.parse_args()


def _load_model(args):
    if not os.path.exists(args.pytorch_ckpt_path):
        print(f"The pth: {args.pytorch_ckpt_path} does not exist. Download pth")

    model = AutoModel.from_pretrained(args.pytorch_ckpt_path, attn_implementation="eager")
    model.eval()

    return model


def _export_text_encoder_onnx(model, args):
    dummy_input_ids = torch.randint(1000, [1000, 3])
    wrapped_model = WrappedSigLIP2TextEncoder(model)
    dynamic_axes = {
        'input_ids': {0: 'labelnums'},
        'text_features': {0: 'labelnums'},
    }
    torch.onnx.export(
        wrapped_model,
        dummy_input_ids,
        os.path.join(args.save_onnx_path, 'siglip2_text_encoder.onnx'),
        input_names=['input_ids'],
        output_names=['text_features'],
        dynamic_axes=dynamic_axes,
        opset_version=15,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=True
    )


def _export_vision_encoder_onnx(model, args):
    dummy_vision_input = torch.rand([2, 3, 384, 384])
    wrapped_model = WrappedSigLIP2VisionEncoder(model)
    dynamic_axes = {
        'image': {0: 'batch_size'},
        'image_features': {0: 'batch_size'},
    }
    torch.onnx.export(
        wrapped_model,
        dummy_vision_input,
        os.path.join(args.save_onnx_path, 'siglip2_vision_encoder.onnx'),
        input_names=['image'],
        output_names=['image_features'],
        dynamic_axes=dynamic_axes,
        opset_version=15,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=True
    )


def main(args):
    model = _load_model(args)
    if not os.path.exists(args.save_onnx_path):
        os.mkdir(args.save_onnx_path)

    if args.convert_text:
        _export_text_encoder_onnx(model, args)
        print(f"SigLIP2 Text Encoder Finished PyTorch to ONNX conversion. Saved to {os.path.join(args.save_onnx_path, 'siglip2_text_encoder.onnx')}")

    if args.convert_vision:
        _export_vision_encoder_onnx(model, args)
        print(f"SigLIP2 Vision Encoder Finished PyTorch to ONNX conversion. Saved to {os.path.join(args.save_onnx_path, 'siglip2_vision_encoder.onnx')}")

        
if __name__ == '__main__':
    main(_parse_args())