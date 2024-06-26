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
import time
import onnxruntime as ort
import numpy as np

_N_MEL = 80
_FRAMES = 3000
_MAX_TOKEN = 224
_HALF_FRAMES = 1500
_KV_NUM = 2


def test(encoder_path, provider, output_names, onnx_inputs, meta=""):
    onnx_model = ort.InferenceSession(
        encoder_path,
        providers=[provider]
    )

    # warmup
    for _ in range(10):
        onnx_model.run(output_names, onnx_inputs)
    # performance test
    num_infer = 100
    start = time.time()
    for _ in range(num_infer):
        onnx_model.run(output_names, onnx_inputs)
    end = time.time()

    print(f"{meta} latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"{meta} throughput: {num_infer / (end - start):.2f} fps")


def test_encoder(args, provider):
    x = np.ones((1, _N_MEL, _FRAMES), dtype=np.float16 if args.use_gpu else np.float32)
    onnx_inputs = {'mel': ort.OrtValue.ortvalue_from_numpy(x)}
    output_names = ['ret']

    test(args.encoder_onnx_path, provider, output_names, onnx_inputs, "Encoder")
    

def test_decoder_prefill(args, provider): 
    assert args.ntokens <= _MAX_TOKEN, f'ntokens can not exceed {_MAX_TOKEN}'
    tokens = np.ones((args.beam_size, args.ntokens), dtype=np.int64)
    audio_features = np.ones((1, _HALF_FRAMES, args.hidden), dtype=np.float16 if args.use_gpu else np.float32)
    pos_embed = np.ones((args.ntokens, args.hidden), dtype=np.float32)
    onnx_inputs = {
        'tokens': ort.OrtValue.ortvalue_from_numpy(tokens),
        'audio_features': ort.OrtValue.ortvalue_from_numpy(audio_features),
        'pos_embed': ort.OrtValue.ortvalue_from_numpy(pos_embed)
    }
    output_names = ["logits", "cache_dyn", "cache_sta"]

    test(args.decoder_prefill_onnx_path, provider, output_names, onnx_inputs, "Decoder prefill")


def test_decoder_decode(args, provider):
    assert args.ntokens <= _MAX_TOKEN, f'ntokens can not exceed {_MAX_TOKEN}'
    tokens = np.ones((args.beam_size, 1), dtype=np.int64)
    pos_embed = np.ones((args.hidden), dtype=np.float32)
    cache_dyn = np.ones(
        (args.nblocks, _KV_NUM, args.beam_size, args.ntokens, args.hidden),
        dtype=np.float16 if args.use_gpu else np.float32
    )
    cache_sta = np.ones(
        (args.nblocks, _KV_NUM, 1, _HALF_FRAMES, args.hidden),
        dtype=np.float16 if args.use_gpu else np.float32
    )
    onnx_inputs = {
        'tokens': ort.OrtValue.ortvalue_from_numpy(tokens), # audio_features onnx导出被折叠
        'pos_embed': ort.OrtValue.ortvalue_from_numpy(pos_embed),
        'cache_dyn': ort.OrtValue.ortvalue_from_numpy(cache_dyn),
        'cache_sta': ort.OrtValue.ortvalue_from_numpy(cache_sta)
    }
    output_names = ["logits", "new_cache_dyn", "new_cache_sta"]

    test(args.decoder_decode_onnx_path, provider, output_names, onnx_inputs, "Decoder decode")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_onnx_path',type=str, default='/tmp/models/encoder.onnx')
    parser.add_argument('--decoder_prefill_onnx_path',type=str, default='/tmp/models/decoder_prefill.onnx')
    parser.add_argument('--decoder_decode_onnx_path',type=str, default='/tmp/models/decoder_decode.onnx')
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--ntokens", type=int, default=100)
    parser.add_argument("--nblocks", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=384)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.use_gpu:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    for func in test_encoder, test_decoder_prefill, test_decoder_decode:
        func(args, provider)


if __name__ == "__main__":
    main()
