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

import torch
import numpy as np

sys.path.append("./")
from models import Wav2Lip

device = 'cpu'


def load_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    
        return checkpoint
    except FileNotFoundError:
        print("No checkpoint found at {}".format(checkpoint_path))
        raise

def main():
    parser = argparse.ArgumentParser(description="Wav2lip model pth to onnx")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mel_batch_shape", type=str, default="1,80,16")
    parser.add_argument("--img_batch_shape", type=str, default="6,96,96")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--onnx_dir", type=str, default="./", required=True)

    args = parser.parse_args()

    model = Wav2Lip()
    print("Load checkpoint from: {}".format(args.checkpoint_path))
    checkpoint = load_checkpoint(args.checkpoint_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    model.eval()

    # export onnx model
    # 1. get mel_batch, img_batch
    mel_batch_shape = list(map(int, args.mel_batch_shape.split(',')))
    img_batch_shape = list(map(int, args.img_batch_shape.split(',')))
    
    mel_batch = torch.ones(args.batch_size, mel_batch_shape[0], mel_batch_shape[1], mel_batch_shape[2],
                           dtype=torch.float32).to(device)
    img_batch = torch.ones(args.batch_size, img_batch_shape[0], img_batch_shape[1], img_batch_shape[2],
                           dtype=torch.float32).to(device)
    # 2. get onnx model
    if not os.path.exists(args.onnx_dir):
        os.makedirs(args.onnx_dir, mode=0o777, exist_ok=False)
    onnx_name = f"{args.onnx_dir}/wav2lip_bs{args.batch_size}.onnx"
    torch.onnx.export(model,
                      (mel_batch, img_batch),
                      onnx_name,
                      input_names=["input1", "input2"],
                      output_names=["output"],
                      opset_version=12,
                      keep_initializers_as_inputs=True)


if __name__ == '__main__':
    main()