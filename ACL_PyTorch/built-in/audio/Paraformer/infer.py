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
import time

import torch
import torch_npu

from torchair_auto_model import TorchairAutoModel


def parse_args():
    parser = argparse.ArgumentParser(description="Paraformer Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        help="Path of pretrained paraformer model"
    )
    parser.add_argument(
        "--model_vad",
        type=str,
        default=None,
        help="Path of pretrained vad model"
    )
    parser.add_argument(
        "--model_punc",
        type=str,
        default=None,
        help="Path of pretrained vad model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        '--data',
        type=str,
        default="speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav",
        help='Path of wav file'
    )
    parser.add_argument(
        '--hotwords',
        type=str,
        default=None,
        help='Hotword.'
    )
    parser.add_argument("--threshold", type=float, default=0.98, help="threshold for cif predictor")
    parser.add_argument('--warmup', type=int, default=3, help="Warm up times")
    parser.add_argument('--device', type=int, default=0, help='Npu Device Id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    device = 'npu:{}'.format(args.device)
    model = TorchairAutoModel(model=args.model_path,
                              vad_model=args.model_vad,
                              punc_model=args.model_punc,
                              batch_size=args.batch_size,
                              cif_threshold=args.threshold,
                              device=device)

    if model.kwargs['model'] == "BiCifParaformer":
        if args.hotwords is not None:
            print(f"Using paraformer long context version, hotwords will not be used")
    else:
        if args.hotwords is None:
            print(f"Using paraformer hotword version, but hotword is not specified, please check your input args")
            exit()
    
    # warm up and compile
    with torch.inference_mode():
        print("Begin warming up.")
        for _ in range(args.warmup):
            _ = model.generate(input_data=args.data, hotword=args.hotwords)
        print("Finish warming up")

        print("Begin inference")
        results, time_stats = model.generate(input_data=args.data, hotword=args.hotwords)
        for res in results:
            print(f"transcription result: {res['text']}")