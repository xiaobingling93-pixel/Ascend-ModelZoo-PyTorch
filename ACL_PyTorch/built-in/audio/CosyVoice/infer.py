# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse
import torch_npu
import torch
import torchaudio
from torch_npu.contrib import transfer_to_npu

import torchair as tng
from ais_bench.infer.interface import InferSession
from torchair.configs.compiler_config import CompilerConfig

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav


def model_ini(args):
    # 初始化模型
    cosyvoice = CosyVoice(args.model_path)
    flow_om = InferSession(0, args.flow)
    speech_om = InferSession(0, args.speech)
    campplus_om = InferSession(0, args.campplus)

    # 设置torchair初始化参数
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.llm.llm.eval()
    cosyvoice.model.llm.llm = torch.compile(cosyvoice.model.llm.llm, dynamic=True, fullgraph=True, backend=npu_backend)

    return cosyvoice, flow_om, speech_om, campplus_om

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="CosyVoice infer")
    parser.add_argument("--model_path", type=str, help="modelpath")
    parser.add_argument('--campplus', type=str, help='campplus om model')
    parser.add_argument('--speech', type=str, help='speech token om model')
    parser.add_argument('--flow', type=str, help='flow  om model')
    parser.add_argument('--warm_up_time', default=2, type=int, help='warm up time')
    parser.add_argument('--infer_count', default=10, type=int, help='infer loop count')
    parser.add_argument('--input_length', default=16000, type=int, help='audio load resample size')
    parser.add_argument('--output_length', default=22050, type=int, help='audio save resample size')
    args = parser.parse_args()

    cosyvoice, flow, speech, campplus = model_ini(args)

    # 输入数据加载
    prompt_txt = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    out_txt = '希望你以后能够做的比我还好嘞。'
    prompt_speech_16k = load_wav('zero_shot_prompt.wav', args.input_length)

    print('warm up start')
    for _ in range(args.warm_up_time):
        out = cosyvoice.inference_zero_shot(prompt_txt, out_txt, prompt_speech_16k, flow, speech, campplus)

    print('warm up end')

    print('infer start')
    s1 = time.time()
    for _ in range(args.infer_count):
        out = cosyvoice.inference_zero_shot(prompt_txt, out_txt, prompt_speech_16k, flow, speech, campplus)
    s2 = time.time()
    print('infer end')
    print('infer averge time cost' + str((s2 - s1) / args.infer_count))
    torchaudio.save('zero_shot.wav', out['tts_speech'], args.output_length)
