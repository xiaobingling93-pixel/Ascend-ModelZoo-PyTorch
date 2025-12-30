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
import torch
import torch_npu

from torchair_auto_model import TorchairAutoModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./paraformer_model",
                        help="path of pretrained paraformer")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size of paraformer")
    parser.add_argument("--data_path", default="./data_aishell/wav/test", type=str,
                        help="directory of AISHELL dataset")
    parser.add_argument("--result_path", default="./aishell_test_rsult.txt", type=str,
                        help="path to save infer result")
    parser.add_argument("--warm_up", default=3, type=int,
                        help="num of warm ups")
    parser.add_argument("--threshold", type=float, default=0.98, help="threshold for cif predictor")
    parser.add_argument('--device', type=int, default=0, help='Npu Device Id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    valid_extensions = [".wav"]
    audio_files = []

    if os.path.isfile(args.data_path):
        if any(args.data_path.endswith(ext) for ext in valid_extensions):
            audio_files.append(args.data_path)
    elif os.path.isdir(args.data_path):
        for root, _, files in os.walk(args.data_path):
            for file in files:
                if any(file.endswith(ext) for ext in valid_extensions):
                    audio_files.append(os.path.join(root, file))
    
    # filter out wav files which is smaller than 1KB
    audio_files = [file for file in audio_files if os.path.getsize(file) >= 1024]
    num_audios = len(audio_files)
    print("num audio files: ", num_audios)

    if len(audio_files) == 0:
        print("There is no valid wav file in data_path.")
        exit()

    # initialize auto model
    device = 'npu:{}'.format(args.device)
    model = TorchairAutoModel(model=args.model_path,
                              batch_size=args.batch_size,
                              cif_threshold=args.threshold,
                              device=device)
        
    # warm up
    print("Begin warming up.")
    for i in range(args.warm_up):
        _ = model.inference_with_asr(input_data=audio_files[:args.batch_size])
        print(f"warm up {i}/{args.warm_up}")
    print("Finishing warming up")

    print("Begin evaluating.")
    results, time_stats = model.inference_with_asr(input_data=audio_files, display_pbar=True)
    print("Average transcription rate: {:.3f}".format(time_stats["avg_trans_rate"]))

    # save results
    with open(args.result_path, 'w') as f:
        for res in results:
            f.write(f"{res['key']} {res['text']}\n")