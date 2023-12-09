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
    parser.add_argument('--need_trace', default="true",
                        required=False, help='If you have traced the model before then set false')
    parser.add_argument('--need_compile', default="true",
                        required=False, help='If you have compiled the model before then set false')

    args = parser.parse_args()
    device = args.device
    batch_size = args.batch_size
    model_path = args.pretrained_model
    need_trace = args.need_trace
    need_compile = args.need_compile
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torchscript=True).float()
    model.eval()


    # stage1: model trace
    if need_trace == "true":
        print("===================== start to trace model ==========================")
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
        traced_model = torch.jit.trace(model, example_kwarg_inputs=input_dict)
        traced_model_path = "./chatglm2_6b_batch" + str(batch_size) + "_traced.pt"
        torch.jit.save(traced_model, traced_model_path)
        print("===================== model trace success ==========================")

    # stage2: model compile
    if need_compile == "true":
        ## load origin traced model
        traced_model_path = "./chatglm2_6b_batch" + str(batch_size) + "_traced.pt"
        try:
            traced_model = torch.jit.load(traced_model_path)
        except Exception as e:
            print("load model failed, please trace first.")

        ## set compile config
        inputs = []
        max_seqlen = 32768
        input0_min_shape = (batch_size, 1)
        input0_max_shape = (batch_size, max_seqlen)
        input1_min_shape = (batch_size, 1)
        input1_max_shape = (batch_size, max_seqlen)
        input2_min_shape = (batch_size, 1)
        input2_max_shape = (batch_size, max_seqlen)
        input3_min_shape = (batch_size, 2, 0, batch_size, 2, 128)
        input3_max_shape = (batch_size, 2, max_seqlen, batch_size, 2, 128)

        inputs.append(torch_aie.Input(min_shape = input0_min_shape, max_shape = input0_max_shape, dtype = torch.int64))
        inputs.append(torch_aie.Input(min_shape = input1_min_shape, max_shape = input1_max_shape, dtype = torch.int64))
        inputs.append(torch_aie.Input(min_shape = input2_min_shape, max_shape = input2_max_shape, dtype = torch.int64))
        inputs.append(torch_aie.Input(min_shape = input3_min_shape, max_shape = input3_max_shape, dtype = torch.float32))

        ## compile
        print("===================== start to compile model ==========================")
        compiled_module = torch_aie.compile(
            traced_model,
            inputs=inputs,
            precision_policy=_enums.PrecisionPolicy.FP32,
            allow_tensor_replace_int=True,
            soc_version="Ascend910B4"
        )
        print("===================== model compile success ==========================")
        ## save compiled result
        aie_model_path = "./chatglm2_6b_batch" + str(batch_size) + "_compiled.ts"
        compiled_module.save(aie_model_path)
        print("===================== save compiled model success ======================")
        

if __name__ == '__main__':
    main()