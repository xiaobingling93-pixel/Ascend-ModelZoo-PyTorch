# Copyright 2025 Huawei Technologies Co., Ltd
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

import argparse
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from model import SenseVoiceSmall


if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="Sensevoice infer")
    parser.add_argument("--model_path", type=str, help="modelpath")
    parser.add_argument('--device', type=int, help='npu device num')
    parser.add_argument('--input', type=str, help='input audio file')
    parser.add_argument('--perform', action='store_true', help='test performance')
    parser.add_argument('--loop', default=10, type=int, help='loop time')
    args = parser.parse_args()

    # 初始化pytorch模型
    m, kwargs = SenseVoiceSmall.from_pretrained(args.model_path)
    m.eval()
    m.half()
    # 设置torchair参数
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backbend = tng.get_npu_backend(compiler_config=config)
    m.encoder = torch.compile(m.encoder, dynamic=True, fullgraph=True, backend=npu_backbend)
    tng.use_internal_format_weight(m.encoder)

    with torch.no_grad():
        # 执行推理
        res, _ = m.inference(
            data_in=args.input,
            language="auto",
            use_itn=False,
            ban_emo_unk=False,
            **kwargs,
        )
        text = rich_transcription_postprocess(res[0]['text'])
        print('语音输出:')
        print(text)

        if args.perform:
            # 执行性能测试
            t = 0
            for _ in range(args.loop):
                res, meta_data = m.inference(
                    data_in=args.input,
                    language="auto",
                    use_itn=False,
                    ban_emo_unk=False,
                    **kwargs,
                )
                t += meta_data["cost_time"]
            print('单条数据推理耗时：')
            print(str(t / args.loop))     
