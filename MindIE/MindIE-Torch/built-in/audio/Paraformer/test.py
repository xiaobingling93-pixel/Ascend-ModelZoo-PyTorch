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

import os
import argparse

import torch
import torch_npu

import mindietorch

from mindie_auto_model import MindieAutoModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model",
                        help="path of pretrained model")
    parser.add_argument("--model_vad", default="./model_vad",
                        help="path of pretrained vad model")
    parser.add_argument("--model_punc", default="./model_punc",
                        help="path of pretrained punc model")
    parser.add_argument("--compiled_encoder", default="./compiled_model/compiled_encoder.pt",
                        help="path to save compiled encoder")
    parser.add_argument("--compiled_decoder", default="./compiled_model/compiled_decoder.pt",
                        help="path to save compiled decoder")
    parser.add_argument("--compiled_cif", default="./compiled_model/compiled_cif.pt",
                        help="path to save compiled cif function")
    parser.add_argument("--compiled_cif_timestamp", default="./compiled_model/compiled_cif_timestamp.pt",
                        help="path to save compiled cif timestamp function")
    parser.add_argument("--compiled_punc", default="./compiled_model/compiled_punc.ts",
                        help="path to save compiled punc model")
    parser.add_argument("--compiled_vad", default="./compiled_model/compiled_vad.ts",
                        help="path to save compiled vad model")
    parser.add_argument("--paraformer_batch_size", default=16, type=int,
                        help="batch size of paraformer model")
    parser.add_argument("--sample_path", default="./audio/",
                        help="directory or path of sample audio")
    parser.add_argument("--soc_version", default="Ascendxxx", type=str,
                        help="soc version of Ascend")
    args = parser.parse_args()

    mindietorch.set_device(0)

    valid_extensions = ['.wav']
    audio_files = []

    if os.path.isfile(args.sample_path):
        if any(args.sample_path.endswith(ext) for ext in valid_extensions):
            audio_files.append(args.sample_path)
    elif os.path.isdir(args.sample_path):
        for root, dirs, files in os.walk(args.sample_path):
            for file in files:
                if any(file.endswith(ext) for ext in valid_extensions):
                    audio_files.append(os.path.join(root, file))

    if len(audio_files) == 0:
        print("There is no valid wav file in sample_dir.")
    else:
        # initialize auto model
        model = MindieAutoModel(model=args.model, vad_model=args.model_vad, punc_model=args.model_punc,
                                compiled_encoder=args.compiled_encoder, compiled_decoder=args.compiled_decoder,
                                compiled_cif=args.compiled_cif, compiled_cif_timestamp=args.compiled_cif_timestamp,
                                compiled_punc=args.compiled_punc, compiled_vad=args.compiled_vad,
                                paraformer_batch_size=args.paraformer_batch_size,
                                cif_interval=200, cif_timestamp_interval=500)

        if "910" in args.soc_version:
            model.kwargs["mindie_device"] = "npu"
        else:
            model.kwargs["mindie_device"] = "cpu"

        # warm up
        print("Begin warming up.")
        _ = model.generate(input_data=audio_files[0])
        print("Finish warming up.")

        # iterate over sample_dir
        for wav_file in audio_files:
            print("\nBegin evaluating {}.".format(wav_file))

            res, time_stats = model.generate(input_data=wav_file)
            print("Model output: {}".format(res[0]["text"]))
            print("Time comsumption:")
            print(" ".join(f"{key}: {value:.3f}s" for key, value in time_stats.items()))