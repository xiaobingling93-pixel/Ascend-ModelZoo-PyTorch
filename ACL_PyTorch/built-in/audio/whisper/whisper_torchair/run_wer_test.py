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
import argparse
from jiwer import wer, Compose, ToUpperCase, RemovePunctuation
import torch
import whisper
import numpy as np

from pipeline import TorchairPipeline, load_audio


def parse_args():
    parser = argparse.ArgumentParser("Whisper wer validation")
    parser.add_argument("--whisper_model_path", type=str, default="./weight/Whisper-large-v3/large-v3.pt",
                        help="whisper model checkpoint file path")
    parser.add_argument("--vad_model_path", type=str, default="./weight/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        help="vad model checkpoint file path")
    parser.add_argument("--speech_path", type=str, default="./LibriSpeech/dev-clean/",
                        help="librispeech dev clean english transaction speech data path")
    parser.add_argument('--device', type=int, default='0', help="npu device id")
    args = parser.parse_args()
    return args


def check_wer(reference, hypothesis):
    preproessor = Compose([
        ToUpperCase(),
        RemovePunctuation()
    ])

    hyp_processed = preproessor(hypothesis)

    error_rate = wer(reference, hyp_processed)
    return error_rate


def get_audio_txt_pairs(root_dir):
    AUDIO_EXTENSIONS = {".flac", ".mp3", ".wav"}
    audio_txt_pairs = []

    for dir_path, _, file_names in os.walk(root_dir):
        if not file_names:
            continue
        txt_files = [f for f in file_names if f.lower().endswith(".txt")]
        txt_file_path = os.path.join(dir_path, txt_files[0])

        transcription_map = {}
        try:
            with open(txt_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    # each line in format of "{index} {text}", e.g.: "1919-142785-0000 ILLUSTRATION LONG PEPPER"
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        print(f"warning: txt file {txt_file_path} line {line_num} format incorrect, missing ground truth text, skip: {line}")
                        continue
                    audio_prefix, transcription_text = parts
                    transcription_map[audio_prefix] = transcription_text
        except Exception as e:
            print(f"Error: reading txt file {txt_file_path} failed, skip. {str(e)}")
            continue

        audio_files = [f for f in file_names if os.path.splitext(f.lower())[1] in AUDIO_EXTENSIONS]
        if not audio_files:
            print(f"warning: directory {dir_path}, no supported audio files found ({AUDIO_EXTENSIONS}), skip this directory")
            continue

        for audio_file in audio_files:
            audio_prefix = os.path.splitext(audio_file)[0]
            if audio_prefix in transcription_map:
                audio_file_path = os.path.join(dir_path, audio_file)
                audio_txt_pairs.append((audio_file_path, transcription_map[audio_prefix]))
            else:
                print(f"warning: no matching audio file {audio_file}")

    return audio_txt_pairs


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('npu:{}'.format(args.device))

    audio_txt_pairs = get_audio_txt_pairs(args.speech_path)
    whisper_decode_options = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=True)

    torchair_pipe = TorchairPipeline(
        whisper_model_path=args.whisper_model_path,
        vad_model_path=args.vad_model_path,
        batch_size=1,
        device_id=args.device,
        whisper_decode_options=whisper_decode_options
    )

    wer_results = []
    total_num_files = len(audio_txt_pairs)
    print(f"total num files: {total_num_files}")
    for idx, (audio_path, reference) in enumerate(audio_txt_pairs):
        print(f"progress: {idx+1} / {total_num_files}")
        audio = load_audio(audio_path)
        result = torchair_pipe.transcribe(audio, batch_size=1)
        error_rate = check_wer(reference, result[0]['text'])
        wer_results.append(error_rate)
    print(f"average wer: {np.mean(wer_results)}")