# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav


if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="CosyVoice infer")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument('--warm_up_times', default=2, type=int, help='warm up times')
    parser.add_argument('--infer_count', default=10, type=int, help='infer loop count')
    args = parser.parse_args()

    cosyvoice = CosyVoice(args.model_path, load_om=True)

    # 设置torchair初始化参数
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.llm.llm.eval()
    cosyvoice.model.llm.llm = torch.compile(cosyvoice.model.llm.llm, dynamic=True, fullgraph=True, backend=npu_backend)

    # 输入数据加载
    prompt_txt = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    out_txt = '希望你以后能够做的比我还好嘞。'

    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000) # 16000为当前语音用例的resample rate参数
    with torch.no_grad():
        print('warm up start')
        for _ in range(args.warm_up_times):
            next(cosyvoice.inference_zero_shot(prompt_txt, out_txt, prompt_speech_16k, stream=False))
        print('warm up end')
        for _ in range(args.infer_count):
            for j in cosyvoice.inference_zero_shot(prompt_txt, out_txt, prompt_speech_16k, stream=False):
                torchaudio.save('zero_shot_result.wav', j['tts_speech'], cosyvoice.sample_rate)