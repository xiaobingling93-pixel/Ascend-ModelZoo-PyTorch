# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import time

import torch
import torch_npu
import torchair as tng
from torchair import CompilerConfig

from funasr import AutoModel


def parse_args():
    parser = argparse.ArgumentParser(description="Paraformer Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path of Pretrained Weight"
    )
    parser.add_argument(
        '--data',
        type=str,
        default="speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/example/asr_example.wav",
        help='Path of wav file'
    )
    parser.add_argument(
        '--hotwords',
        type=str,
        default="魔搭",
        help='Hotword.'
    )
    parser.add_argument('--warmup', type=int, default=4, help="Warm up times")
    parser.add_argument('--device', type=int, default=0, help='Npu Device Id')
    parser.add_argument('--loop', default=10, type=int, help='Loop Times')
    args = parser.parse_args()
    return args


def create_model(args):
    device = torch.device('npu:{}'.format(args.device))
    model = AutoModel(model=args.model_path, disable_update=True)
    model.model.to(device)

    # adapt torchair
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backbend = tng.get_npu_backend(compiler_config=config)
    model.model.encoder = torch.compile(model.model.encoder, dynamic=True, fullgraph=True, backend=npu_backbend)
    tng.use_internal_format_weight(model.model.encoder)
    model.model.decoder = torch.compile(model.model.decoder, dynamic=True, fullgraph=True, backend=npu_backbend)
    tng.use_internal_format_weight(model.model.decoder)

    return model

if __name__ == '__main__':

    args = parse_args()

    model = create_model(args)

    # compile and precision test
    device = torch.device('npu:{}'.format(args.device))
    print('start compiling...')
    res = model.generate(input=args.data, hotword=args.hotwords, device=device)
    print('result:', res)

    # test model performance
    with torch.inference_mode():
        for _ in range(args.warmup):
            model.generate(input=args.data, hotword=args.hotwords, device=device)

        total_time = 0
        for _ in range(args.loop):
            torch.npu.synchronize()
            start = time.time()
            model.generate(input=args.data, hotword=args.hotwords, device=device)
            torch.npu.synchronize()
            total_time += time.time() - start
        print(f'E2E performance = {total_time / args.loop} s/data')
   