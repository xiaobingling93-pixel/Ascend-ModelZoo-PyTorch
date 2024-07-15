# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindietorch

_FRAMES = 3000
_HALF_FRAMES = 1500
_MAX_TOKEN = 224
_KV_NUM = 2

def parse_args():
    parser = argparse.ArgumentParser(description="mindietorch model compilation")
    parser.add_argument("--model_path", default="/tmp/models")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--nblocks", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--soc_version", default="Ascend310P3")
    args = parser.parse_args()
    return args

def compile_and_save(ts_model, input_info, soc_version, save_path):
    ts_model.eval()
    mindie_model = mindietorch.compile(
        ts_model,
        inputs=input_info,
        precision_policy=mindietorch._enums.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        allow_tensor_replace_int=True,
        soc_version=soc_version,
        optimization_level=0
    )
    mindie_model.save(save_path)

def encoder(args):
    ts_model = torch.jit.load(f"{args.model_path}/encoder.ts")
    input_mel_info = mindietorch.Input([1, args.n_mels, _FRAMES])
    input_info = [input_mel_info]
    save_path = f"{args.model_path}/encoder_compiled.ts"
    compile_and_save(ts_model, input_info, args.soc_version, save_path)

def language(args):
    ts_model = torch.jit.load(f"{args.model_path}/decoder_prefill.ts")
    input_tokens_info = mindietorch.Input([1, 1])
    input_audio_features_info = mindietorch.Input([1, _HALF_FRAMES, args.hidden])
    input_pos_embed_info = mindietorch.Input([1, args.hidden])
    input_info = [
        input_tokens_info,
        input_audio_features_info,
        input_pos_embed_info,
    ]
    save_path = f"{args.model_path}/language_detection_compiled.ts"
    compile_and_save(ts_model, input_info, args.soc_version, save_path)

def prefill(args):
    ts_model = torch.jit.load(f"{args.model_path}/decoder_prefill.ts")

    input_tokens_info = mindietorch.Input(
        min_shape=[args.beam_size, 1],
        max_shape=[args.beam_size, _MAX_TOKEN]
    )
    input_audio_features_info = mindietorch.Input(
        min_shape=[1, _HALF_FRAMES, args.hidden],
        max_shape=[1, _HALF_FRAMES, args.hidden]
    )
    input_pos_embed_info = mindietorch.Input(
        min_shape=[1, args.hidden],
        max_shape=[_MAX_TOKEN, args.hidden]
    )
    input_info = [
        input_tokens_info,
        input_audio_features_info,
        input_pos_embed_info,
    ]
    save_path = f"{args.model_path}/decoder_prefill_compiled.ts"
    compile_and_save(ts_model, input_info, args.soc_version, save_path)

def decode(args):
    ts_model = torch.jit.load(f"{args.model_path}/decoder_decode.ts")

    input_tokens_info = mindietorch.Input(
        min_shape=[args.beam_size, 1],
        max_shape=[args.beam_size, 1]
    )
    input_audio_features_info = mindietorch.Input(
        min_shape=[1, _HALF_FRAMES, args.hidden],
        max_shape=[1, _HALF_FRAMES, args.hidden]
    )
    input_pos_embed_info = mindietorch.Input(
        min_shape=[args.hidden],
        max_shape=[args.hidden]
    )
    input_cache_dyn_info = mindietorch.Input(
        min_shape=(args.nblocks, _KV_NUM, args.beam_size, 1, args.hidden),
        max_shape=(args.nblocks, _KV_NUM, args.beam_size, _MAX_TOKEN, args.hidden)
    )
    input_cache_sta_info = mindietorch.Input(
        min_shape=[args.nblocks, _KV_NUM, 1, _HALF_FRAMES, args.hidden],
        max_shape=[args.nblocks, _KV_NUM, 1, _HALF_FRAMES, args.hidden]
    )

    input_info = [
        input_tokens_info,
        input_audio_features_info,
        input_pos_embed_info,
        input_cache_dyn_info,
        input_cache_sta_info
    ]

    save_path = f"{args.model_path}/decoder_decode_compiled.ts"
    compile_and_save(ts_model, input_info, args.soc_version, save_path)

def main():
    args = parse_args()
    for func in encoder, language, prefill, decode:
        func(args)

if __name__ == '__main__':
    main()