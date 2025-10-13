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
import string
import numpy as np
import zhconv
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import whisper

from modeling_whisper import get_whisper_model
from pipeline import load_audio, SAMPLE_RATE


def parse_args():
    parser = argparse.ArgumentParser("Whisper transcribe")
    parser.add_argument("--whisper_model_path", type=str, default="./weight/Whisper-large-v3/large-v3.pt",
                        help="whisper model checkpoint file path")
    parser.add_argument("--language", type=str, default='zh', help="output language")
    parser.add_argument("--sample_audio", type=str, default="./audio.mp3",
                        help="sample audio for warm up and compilation")
    parser.add_argument("--audio_path", type=str, required=True, help="target audio to transcribe")
    parser.add_argument("--device", type=int, default='0', help="npu device id")
    parser.add_argument("--warmup", type=int, default='3', help="warm up times")
    return parser.parse_args()


def generateSubtitles(model, file_path, language='zh'):
    audio = load_audio(file_path, SAMPLE_RATE)

    print("start transcribe...")
    output = model.transcribe(audio, language=language)
    segments_list = output['segments']

    text = ''
    index = 1
    last_sentence = None
    segment_lines = []
    comma = ', '
    full_stop = '.'
    if language == 'zh':
        comma = '，'
        full_stop = '。'

    for segment in segments_list:
        if segment['temperature'] > 0.8:
            print(f"Low confidence segment detected, skipping... {segment['text']}")
            continue
        sentence = segment['text']
        if sentence.strip() in {'', '.', ',', '，', '。'}:
            continue
        if language == 'zh' or language == 'jw':
            # 将繁体字转化成简体
            sentence = zhconv.convert(sentence, 'zh-cn')
        if last_sentence is not None and last_sentence == sentence:
            continue

        segment_lines.append({
            'index': index,
            'startTime': float(segment['start']),
            'endTime': float(segment['end']),
            'sentence': sentence
        })
        index += 1
        
        # 添加标点符号
        if sentence[-1] not in set(string.punctuation + "，。？！；："):
            sentence += comma
        text += sentence
    
    # 最后一个逗号替换成句号
    if len(text.strip()) > 0 and text[:-1] == comma:
        text = text[:-1] + full_stop
    
    result = {"text": text, "segment_lines": segment_lines}
    print(result)
    with open("result.txt", "w") as f:
        f.write(f"{result}")
    

if __name__ == '__main__':
    args = parse_args()
    deivce = torch.device('npu:{}'.format(args.device))
    whisper_decode_options = whisper.DecodingOptions(without_timestamps=True, fp16=True)
    whisper_model = get_whisper_model(args.whisper_model_path, whisper_decode_options, deivce)

    torch_npu.npu.set_compile_mode(jit_compile=False)
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True  # 使能tiling全下沉配置
    npu_backend = tng.get_npu_backend(compiler_config=config)

    print("compile model...")
    whisper_model.encoder.forward = torch.compile(whisper_model.encoder.forward, dynamic=False, fullgraph=True, backend=npu_backend)
    whisper_model.prefill_decoder.forward = torch.compile(whisper_model.prefill_decoder.forward, dynamic=True, fullgraph=True, backend=npu_backend)
    whisper_model.decode_decoder.forward = torch.compile(whisper_model.decode_decoder.forward, dynamic=True, fullgraph=True, backend=npu_backend)

    sample_audio = load_audio(args.sample_audio)
    print("start warm up...")
    for i in range(args.warmup):
        result = whisper_model.transcribe(sample_audio, language='zh')
        print(f"warm up {i}/{args.warmup} {result['text']}")
    print("warm up done")

    generateSubtitles(whisper_model, args.audio_path, args.language)