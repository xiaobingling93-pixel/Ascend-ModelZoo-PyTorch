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

import os
import time
import math
import argparse

import torch
import torch_npu
import numpy as np
import whisper
import librosa
from jiwer import wer, Compose, ToUpperCase, RemovePunctuation

from pipeline import TorchairPipeline, load_audio
from run_wer_test import check_wer


def parse_args():
    parser = argparse.ArgumentParser("Whisperx infer")
    parser.add_argument("--whisper_model_path", type=str, default="./weight/Whisper-large-v3/large-v3.pt",
                        help="whisper model checkpoint file path")
    parser.add_argument("--vad_model_path", type=str, default="./weight/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        help="vad model checkpoint file path")
    parser.add_argument("--audio_path", type=str, default="./audio.mp3",
                        help="audio file path")
    parser.add_argument("--librispeech_perf_test", dest="librispeech_perf_test", action="store_true",
                        help="conduct performance test on LibriSpeech dataset")
    parser.add_argument("--skip_librispeech_perf_test", dest="librispeech_perf_test", action="store_false",
                        help="skip performance test on LibriSpeech dataset")
    parser.set_defaults(librispeech_perf_test=True)
    parser.add_argument("--speech_path", type=str, default="./LibriSpeech/dev-clean/",
                        help="librispeech dev clean english transaction speech data path")
    parser.add_argument("--num_audio_files", type=int, default=52,
                        help="num of audio files selected from LibriSpeech for performance test")
    parser.add_argument("--librispeech_wer_demo", dest="librispeech_wer_demo", action="store_true",
                        help="perform librispeech wer demo")
    parser.add_argument("--skip_librispeech_wer_demo", dest="librispeech_wer_demo", action="store_false",
                        help="skip librispeech wer demo")
    parser.set_defaults(librispeech_wer_demo=True)
    parser.add_argument('--device', type=int, default='0', help="npu device id")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--warmup', type=int, default=4, help="Warm up times")
    args = parser.parse_args()
    return args


def collect_audio_files(paths: list, extensions: list = None) -> list:
    if extensions is None:
        extensions = ['wav', 'mp3', 'flac']
    extensions = [ext.lower().lstrip('.') for ext in extensions]

    audio_files = []
    for path in paths:
        if not os.path.exists(path):
            print(f"warning: path not exists - {path}")
            continue

        if os.path.isfile(path):
            file_ext = os.path.splitext(path)[1].lower().lstrip('.')
            if file_ext in extensions:
                audio_files.append(os.path.abspath(path))
        
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower().lstrip('.')
                    if file_ext in extensions:
                        audio_path = os.path.join(root, file)
                        audio_files.append(os.path.abspath(audio_path))
    return audio_files


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('npu:{}'.format(args.device))
    whisper_decode_options = whisper.DecodingOptions(without_timestamps=True, fp16=True)

    torchair_pipe = TorchairPipeline(
        whisper_model_path=args.whisper_model_path,
        vad_model_path=args.vad_model_path,
        batch_size=args.batch_size,
        device_id=args.device,
        whisper_decode_options=whisper_decode_options
    )

    audio_sample = load_audio(args.audio_path)

    with torch.inference_mode():
        for _step in range(args.warmup):
            result = torchair_pipe.transcribe(audio_sample, batch_size=args.batch_size)
            print(f"warm up {_step}/{args.warmup} {result[0]['text']}")
        print(f"warm up success.")

        # 对输入的audio进行推理
        y, audio_sr = librosa.load(args.audio_path)
        duration_seconds = librosa.get_duration(y=y, sr=audio_sr)
        print(f"perform inference on input audio: {args.audio_path}")
        t0 = time.time()
        result = torchair_pipe.transcribe(audio_sample, batch_size=args.batch_size)
        t1 = time.time()
        print(f"transcription result: {result}")
        print(f"transcription ratio: {duration_seconds / (t1 - t0)}, speech durarations {duration_seconds}")

        if args.librispeech_perf_test:
            # 对Librispeech数据集中的音频进行推理
            data_path = f'{args.speech_path}/1919/142785'
            audio_files = collect_audio_files([data_path])[:args.num_audio_files]
            if len(audio_files) == 0:
                print(f"No valid audio files found in {data_path}, please check speech_path: {args.speech_path}")
                exit()
            speech_data_list = list(map(load_audio, audio_files))
            speech_data = np.concatenate(speech_data_list)
            duration_seconds = 0
            for audio in audio_files:
                y, audio_sr = librosa.load(audio)
                duration_seconds += librosa.get_duration(y=y, sr=audio_sr)

            print(f"------performance test on samples from LibriSpeech dataset-----")
            t0 = time.time()
            result = torchair_pipe.transcribe(speech_data, batch_size=args.batch_size)
            t1 = time.time()
            print(f"LibriSpeech transcription result: {result}")
            print(f"transcription ratio: {duration_seconds / (t1 - t0)}, speech durarations {duration_seconds}")

        # wer test
        if args.librispeech_wer_demo:
            sample = load_audio(f'{args.speech_path}/1919/142785/1919-142785-0007.flac')
            result = torchair_pipe.transcribe(sample, batch_size=args.batch_size)
            
            reference = "MODE CHOOSE THE GREENEST CUCUMBERS AND THOSE THAT ARE MOST FREE FROM SEEDS \
                    PUT THEM IN STRONG SALT AND WATER WITH A CABBAGE LEAF TO KEEP THEM DOWN TIE A PAPER OVER \
                    THEM AND PUT THEM IN A WARM PLACE TILL THEY ARE YELLOW THEN WASH THEM AND SET THEM OVER THE \
                    FIRE IN FRESH WATER WITH A VERY LITTLE SALT AND ANOTHER CABBAGE LEAF OVER THEM COVER VERY CLOSELY BUT TAKE CARE THEY DO NOT BOIL"
            
            print(f"perform wer demo on a single audio sample from LibriSpeech dataset: {args.speech_path}/1919/142785/1919-142785-0007.flac")
            error_rate = check_wer(reference, result[0]['text'])
            print(f"wer: {error_rate:.4f}")
