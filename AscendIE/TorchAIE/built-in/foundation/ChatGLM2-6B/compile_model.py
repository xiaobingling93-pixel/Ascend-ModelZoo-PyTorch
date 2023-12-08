# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
from transformers import AutoTokenizer, AutoModel
import torch
import torch_aie
from torch_aie import _enums
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='./model/',
                        type=str, required=False, help='model checkpoint path')
    parser.add_argument('--batch_size', default=1, type=int,
                        required=False, help='batch size')
    parser.add_argument('--device', default=0, type=int,
                        required=False, help='npu device')

    args = parser.parse_args()
    device = args.device
    batch_size = args.batch_size
    model_path = args.pretrained_model
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torchscript=True).float()
    model.eval()

    ## set input
    input_ids = torch.randint(1, 64970, [1, 128], dtype = torch.int64)
    position_ids = model.get_position_ids(input_ids, device = "cpu")
    attention_mask = torch.ones((1, 128), dtype = torch.int64)
    past_key_values = torch.rand([28, 2, 0, 1, 2, 128], dtype = torch.float32)
    input_dict = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values
    }

    aie_model_path = "chatglm2_6b_" + str(batch_size) + ".ts"

    torch_aie.set_device(device)

    with torch.inference_mode():
        jit_model = torch.jit.trace(model, example_kwarg_inputs=input_dict)
        aie_input_spec = [torch_aie.Input(
            accept_size, dtype=torch_aie.dtype.INT64),]
        aie_model = torch_aie.compile(
            jit_model,
            inputs=[torch_aie.Input([1, 128], dtype = torch.int64),
                    torch_aie.Input([1, 128], dtype = torch.int64),
                    torch_aie.Input([1, 128], dtype = torch.int64),
                    torch_aie.Input([28, 2, 0, 1, 2, 128], dtype = torch.float64),
                    ],
            precision_policy=_enums.PrecisionPolicy.FP32,
            allow_tensor_replace_int=True,
            soc_version="Ascend910B4")
        aie_model.save(aie_model_path)

if __name__ == '__main__':
    main()