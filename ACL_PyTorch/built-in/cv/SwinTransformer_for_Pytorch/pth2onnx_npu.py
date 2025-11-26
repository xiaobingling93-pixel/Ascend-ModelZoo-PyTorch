# Copyright 2025 Huawei Technologies Co., Ltd
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
import torch
import timm


def pth2onnx(args):
    pth_path = args.model_path
    batch_size = args.batch_size
    out_path = args.out_path

    checkpoint = torch.load(pth_path, map_location='cpu')
    config = checkpoint['config']
    state_dict = checkpoint['model']

    model_name = config.MODEL.NAME

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=config.MODEL.NUM_CLASSES
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    window_size = None
    if hasattr(config.MODEL, 'SWIN') and hasattr(config.MODEL.SWIN, 'WINDOW_SIZE'):
        window_size = config.MODEL.SWIN.WINDOW_SIZE
    elif hasattr(config.MODEL, 'WINDOW_SIZE'):
        window_size = config.MODEL.WINDOW_SIZE
    
    if window_size is None:
        raise ValueError("Error: Unable to get window size from model config, please check config structure")
    M = window_size * window_size

    # Fix shape mismatch for relative_position_index
    for key in list(new_state_dict.keys()):
        if 'relative_position_index' in key:
            tensor_value = new_state_dict.get(key)
            if tensor_value is not None and tensor_value.shape == torch.Size([M * M]):
                new_state_dict[key] = tensor_value.view(M, M)

    model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()

    input_size = None
    # Get input size
    if hasattr(config, 'DATA') and hasattr(config.DATA, 'IMG_SIZE'):
        input_size = config.DATA.IMG_SIZE
    elif 's3' in model_name:
        input_size = int(model_name.split('_')[3])
    else:
        input_size = int(model_name.split('_')[4])
       
    if input_size is None:
        raise ValueError("Error: Unable to get input_size from model config, please check config structure")
    
    input_data = torch.randn([batch_size, 3, input_size, input_size], dtype=torch.float32)

    print("Start exporting ONNX...")

    # Export ONNX
    torch.onnx.export(
        model,
        input_data,
        out_path,
        verbose=True,
        opset_version=11,
        input_names=["image"],
        output_names=["output"],
    )

    print(f"ONNX model saved to: {out_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='SwinTransformer onnx export.')
    parser.add_argument('-i', '--model_path', type=str, required=True,
                        help='model_path for pth model')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='save path for output onnx model')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size for output model')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    pth2onnx(args)