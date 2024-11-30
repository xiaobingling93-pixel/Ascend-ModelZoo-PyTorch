# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
import argparse

import torch
import mindietorch
from mindietorch import _enums

def export_mindietorch(opt):
    trace_model = torch.jit.load(opt.torch_script_path)
    trace_model.eval()

    mindietorch.set_device(0)
    inputs = []
    if opt.export_part == 'encoder':
        inputs.append(mindietorch.Input([opt.batch_size, 100, 80], dtype = mindietorch.dtype.FLOAT))
        inputs.append(mindietorch.Input([opt.batch_size], dtype = mindietorch.dtype.INT64))
    elif opt.export_part == 'decoder':
        inputs.append(mindietorch.Input([opt.batch_size, 2], dtype=mindietorch.dtype.INT64))
    else:
        inputs.append(mindietorch.Input([opt.batch_size, 512], dtype=mindietorch.dtype.FLOAT))
        inputs.append(mindietorch.Input([opt.batch_size, 512], dtype=mindietorch.dtype.FLOAT))

    torchaie_model = mindietorch.compile(
        trace_model,
        inputs=inputs,
        precision_policy=_enums.PrecisionPolicy.FP32,
        soc_version=opt.soc_version,
        optimization_level=0
    )
    suffix = os.path.splitext(opt.torch_script_path)[-1]
    saved_name = os.path.basename(opt.torch_script_path).split('.')[0] + f"_mindietorch_bs{opt.batch_size}" + suffix
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    torchaie_model.save(os.path.join(opt.save_path, saved_name))
    print("torch aie tdnn compiled done. saved model is ", os.path.join(opt.save_path, saved_name))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_script_path', type=str, default='../egs/librispeech/ASR/icefall-asr-zipformer-wenetspeech-20230615/exp/encoder-epoch-12-avg-1.pt', help='trace model path')
    parser.add_argument('--export_part', type=str, default='encoder', help='the part of model(encoder, decoder, and joiner) to be exported.')
    parser.add_argument('--soc_version', type=str, required=True, help='soc version')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_path', type=str, default='./pt_compiled_model/', help='compiled model path')
    opt = parser.parse_args()
    return opt

def main(opt):
    export_mindietorch(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)