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

import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import whisper
from modeling_whisper import get_whisper_model


def parse_args():
    parser = argparse.ArgumentParser("Whisper infer")
    parser.add_argument("--whisper_model_path", type=str, default="./weight/Whisper-large-v3/large-v3.pt",
                        help="whisper model checkpoint file path")
    parser.add_argument("--audio_path", type=str, default="./audio.mp3",
                        help="audio file path")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--warmup', type=int, default=3, help="Warm up times")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    whisper_decode_options = whisper.DecodingOptions(without_timestamps=True, fp16=True)
    whisper_model = get_whisper_model(args.whisper_model_path, whisper_decode_options)

    torch_npu.npu.set_compile_mode(jit_compile=False)
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True  # 使能tiling全下沉配置
    npu_backend = tng.get_npu_backend(compiler_config=config)

    print("compile model...")
    whisper_model.encoder.forward = torch.compile(whisper_model.encoder.forward, dynamic=False, fullgraph=True, backend=npu_backend)
    whisper_model.prefill_decoder.forward = torch.compile(whisper_model.prefill_decoder.forward, dynamic=True, fullgraph=True, backend=npu_backend)
    whisper_model.decode_decoder.forward = torch.compile(whisper_model.decode_decoder.forward, dynamic=True, fullgraph=True, backend=npu_backend)

    audio = whisper.load_audio(args.audio_path)
    audio = whisper.pad_or_trim(audio)
    audio_mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)
    audio_mel = audio_mel.unsqueeze(0).repeat(args.batch_size, 1, 1)
    with torch.inference_mode():
        print("start warm up...")
        for i in range(args.warmup):
            result = whisper.decode(whisper_model, audio_mel, whisper_decode_options)
            print(f"warm up {i}/{args.warmup}")
        print("warm up done")

        print("start inference...")
        result = whisper.decode(whisper_model, audio_mel, whisper_decode_options)
        result_txt = [res.text for res in result]
        print(f"transcription result: {result_txt}")
