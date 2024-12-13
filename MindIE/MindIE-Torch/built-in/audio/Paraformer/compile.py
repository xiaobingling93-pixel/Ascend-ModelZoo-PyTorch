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

import argparse

import torch
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
    parser.add_argument("--compiled_punc", default="./compiled_model/compiled_punc.pt",
                        help="path to save compiled punc model")
    parser.add_argument("--compiled_vad", default="./compiled_model/compiled_vad.pt",
                        help="path to save compiled punc model")
    parser.add_argument("--traced_encoder", default=None,
                        help="path to save traced encoder model")
    parser.add_argument("--traced_decoder", default=None,
                        help="path to save traced decoder model")
    parser.add_argument("--soc_version", required=True, type=str,
                        help="soc version of Ascend")
    args = parser.parse_args()

    mindietorch.set_device(0)

    # use mindietorch to compile sub-models in Paraformer
    print("Begin compiling sub-models.")
    MindieAutoModel.export_model(model=args.model_vad, compiled_path=args.compiled_vad,
                                 compile_type="vad", soc_version=args.soc_version)
    MindieAutoModel.export_model(model=args.model_punc, compiled_path=args.compiled_punc,
                                 compile_type="punc", soc_version=args.soc_version)
    MindieAutoModel.export_model(model=args.model, compiled_encoder=args.compiled_encoder,
                                 compiled_decoder=args.compiled_decoder, compiled_cif=args.compiled_cif,
                                 compiled_cif_timestamp=args.compiled_cif_timestamp, 
                                 traced_encoder=args.traced_encoder, traced_decoder=args.traced_decoder,
                                 cif_interval=200, cif_timestamp_interval=500,
                                 compile_type="paraformer", soc_version=args.soc_version)
    print("Finish compiling sub-models.")