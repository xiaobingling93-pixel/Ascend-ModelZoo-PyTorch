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
import time
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cosyvoice.cli.cosyvoice import CosyVoice2


def no_stream_input_inference(args, cosyvoice, prompt_txt):
    with torch.no_grad():
        print('warm up start')
        for _ in range(args.warm_up_times):
            for _ in enumerate(cosyvoice.inference_sft(prompt_txt[0], '中文女', stream=args.stream_out)):
                pass
        print('warm up end')
        infer_res = [torch.tensor([]) for _ in range(args.infer_count)]
        rtf = []
        for i_step in range(args.infer_count):
            start_time = time.time()
            for _, j in enumerate(cosyvoice.inference_sft(prompt_txt[0], '中文女', stream=args.stream_out)):
                infer_res[i_step] = torch.cat((infer_res[i_step], j['tts_speech']), dim=1)
            end_time = time.time()
            speech_len = infer_res[i_step].shape[1] / cosyvoice.sample_rate
            print(f"singe infer RTF: {(end_time - start_time) / speech_len}")
            rtf.append((end_time - start_time) / speech_len)
            print(f"save out wav file to sft_out_{i_step+1}.wav")
            torchaudio.save(f"sft_out_{i_step+1}.wav", infer_res[i_step], cosyvoice.sample_rate)
        print(f"avg RTF: {sum(rtf) / len(rtf)}")


def stream_input_inference(args, cosyvoice, prompt_txt):

    def inference_step(step, mode):
        times = args.warm_up_times if mode == "warmup" else args.infer_count
        print(f"第{step + 1}/{times}轮 {mode}：↓↓↓")
        print(f"curr prompt text：{prompt_txt[step % len(prompt_txt)]}")
        for char_idx, char in enumerate(prompt_txt[step % len(prompt_txt)]):
            if char_idx == len(prompt_txt[step % len(prompt_txt)]) - 1:
                for _, j in enumerate(cosyvoice.inference_sft_streaming_input(char, char_idx, "中文女", user_id="AscendUser", input_end=True, stream=args.stream_out)):
                    if mode == "warmup":
                        pass
                    else:
                        infer_res[step] = torch.cat((infer_res[step], j['tts_speech']), dim=1)
            else:
                for _, j in enumerate(cosyvoice.inference_sft_streaming_input(char, char_idx, "中文女", user_id="AscendUser", input_end=False, stream=args.stream_out)):
                    if mode == "warmup":
                        pass
                    else:
                        infer_res[step] = torch.cat((infer_res[step], j['tts_speech']), dim=1)

    infer_res = [torch.tensor([]) for _ in range(args.infer_count)]

    with torch.no_grad():
        print("warm up start")
        for w_step in range(args.warm_up_times):
            inference_step(w_step, mode="warmup")
        print("warm up end")

        print("inference start")
        rtf = []
        for i_step in range(args.infer_count):
            start_time = time.time()
            inference_step(i_step, mode="inference")
            end_time = time.time()
            speech_len = infer_res[i_step].shape[1] / cosyvoice.sample_rate
            print(f"avg RTF: {(end_time - start_time) / speech_len}")
            rtf.append((end_time - start_time) / speech_len)
            print(f"save out wav file to stream_input_out_{i_step+1}.wav")
            torchaudio.save(f"stream_input_out_{i_step+1}.wav", infer_res[i_step], cosyvoice.sample_rate)
        print(f"avg RTF: {sum(rtf) / len(rtf)}")
        print("inference end")


if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)

    parser = argparse.ArgumentParser(description="CosyVoice2 infer")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument('--warm_up_times', default=2, type=int, help='warm up times')
    parser.add_argument('--infer_count', default=20, type=int, help='infer loop count')
    parser.add_argument('--stream_in', action="store_true", help='stream input infer')
    parser.add_argument('--stream_out', action="store_true", help='stream output infer')
    args = parser.parse_args()

    cosyvoice = CosyVoice2(args.model_path, load_om=True, fp16=True)
    cosyvoice.model.llm.eval()
    cosyvoice.model.llm.llm.model.model.half()

    # 对hift模型结构进行torchair图模式适配
    cosyvoice.model.hift.remove_weight_norm()
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    cosyvoice.model.hift.decode = torch.compile(cosyvoice.model.hift.decode, dynamic=True, fullgraph=True, backend=npu_backend)

    # 输入数据加载
    prompt_txt = [
        '收到好友从远方寄来的生日礼物，那份意外的惊喜和深深的祝福，让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
        '全球每年有超过一百三十五万人，因吸烟而死亡'
    ]

    # 普通输入（非流式输入）
    if not args.stream_in:
        no_stream_input_inference(args, cosyvoice, prompt_txt)
    # 流式输入
    else:
        stream_input_inference(args, cosyvoice, prompt_txt)
