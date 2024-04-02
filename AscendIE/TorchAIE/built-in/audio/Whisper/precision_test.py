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
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import mindietorch

_N_MEL = 80
_FRAMES = 3000
_MAX_TOKEN = 224
_HALF_FRAMES = 1500
_HIDDEN = 384
_KV_NUM = 2

def compare_onnx_aie_output(onnx_out, aie_out, sim_threshold=0.99):
    num_sim = 0
    for i, (a, b) in enumerate(zip(onnx_out, aie_out)):
        a = a.reshape(1, -1).astype(np.float32)
        b = b.reshape(1, -1)
        sim = F.cosine_similarity(torch.from_numpy(a), b, dim=1)
        if sim > sim_threshold:
            num_sim += 1
        else:
            print(f'Output {i} similarity: {sim}')

    print(f'Number of outputs to compare: {len(onnx_out)}')
    print(f'Number of outputs with cosine similarity > {sim_threshold}: {num_sim}')


def compare_encoder(args):
    device = f'npu:{args.device_id}'

    onnx_model = ort.InferenceSession(
        args.encoder_onnx_path,
        providers=["CPUExecutionProvider"]
    )

    x = np.ones((1, _N_MEL, _FRAMES), dtype=np.float32)
    onnx_inputs = {'mel': ort.OrtValue.ortvalue_from_numpy(x)}
    output_names = ['ret']
    onnx_out = onnx_model.run(output_names, onnx_inputs)

    aie_inputs = [x]
    for i in range(len(aie_inputs)):
        aie_inputs[i] = torch.from_numpy(aie_inputs[i]).to(device)
    
    mindietorch.set_device(args.device_id)
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(args.encoder_aie_path)
    model.eval().to(device)

    with mindietorch.npu.stream(stream):
        aie_out = model(*aie_inputs)
        stream.synchronize()
    
    if isinstance(aie_out, tuple):
        aie_out = (x.cpu() for x in aie_out)
    else:
        aie_out = aie_out.cpu()
    compare_onnx_aie_output(onnx_out, aie_out, args.sim_threshold)


def compare_decoder_prefill(args):
    device = f'npu:{args.device_id}'

    onnx_model = ort.InferenceSession(
        args.decoder_prefill_onnx_path,
        providers=["CPUExecutionProvider"]
    )

    assert args.ntokens <= _MAX_TOKEN, f'ntokens can not exceed {_MAX_TOKEN}'
    tokens = np.ones((args.beam_size, args.ntokens), dtype=np.int64)
    audio_features = np.ones((1, _HALF_FRAMES, _HIDDEN), dtype=np.float32)
    pos_embed = np.ones((args.ntokens, _HIDDEN), dtype=np.float32)
    onnx_inputs = {
        'tokens': ort.OrtValue.ortvalue_from_numpy(tokens),
        'audio_features': ort.OrtValue.ortvalue_from_numpy(audio_features),
        'pos_embed': ort.OrtValue.ortvalue_from_numpy(pos_embed)
    }
    output_names = ["logits", "cache_dyn", "cache_sta"]
    onnx_out = onnx_model.run(output_names, onnx_inputs)

    aie_inputs = [tokens.astype(np.float32), audio_features, pos_embed]
    for i in range(len(aie_inputs)):
        aie_inputs[i] = torch.from_numpy(aie_inputs[i]).to(device)
    
    mindietorch.set_device(args.device_id)
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(args.decoder_prefill_aie_path)
    model.eval().to(device)

    with mindietorch.npu.stream(stream):
        aie_out = model(*aie_inputs)
        stream.synchronize()
    if isinstance(aie_out, tuple):
        aie_out = (x.cpu() for x in aie_out)
    else:
        aie_out = aie_out.cpu()
    compare_onnx_aie_output(onnx_out, aie_out, args.sim_threshold)


def compare_decoder_decode(args):
    device = f'npu:{args.device_id}'

    onnx_model = ort.InferenceSession(
        args.decoder_decode_onnx_path,
        providers=["CPUExecutionProvider"]
    )

    assert args.ntokens <= _MAX_TOKEN, f'ntokens can not exceed {_MAX_TOKEN}'
    tokens = np.ones((args.beam_size, 1), dtype=np.int64)
    audio_features = np.ones((1, _HALF_FRAMES, _HIDDEN), dtype=np.float32)
    pos_embed = np.ones((_HIDDEN), dtype=np.float32)
    cache_dyn = np.ones((args.nblocks, _KV_NUM, args.beam_size, args.ntokens, _HIDDEN), dtype=np.float32)
    cache_sta = np.ones((args.nblocks, _KV_NUM, 1, _HALF_FRAMES, _HIDDEN), dtype=np.float32)
    onnx_inputs = {
        'tokens': ort.OrtValue.ortvalue_from_numpy(tokens), # audio_features onnx导出被折叠
        'pos_embed': ort.OrtValue.ortvalue_from_numpy(pos_embed),
        'cache_dyn': ort.OrtValue.ortvalue_from_numpy(cache_dyn),
        'cache_sta': ort.OrtValue.ortvalue_from_numpy(cache_sta)
    }

    output_names = ["logits", "new_cache_dyn", "new_cache_sta"]
    onnx_out = onnx_model.run(output_names, onnx_inputs)

    aie_inputs = [tokens.astype(np.float32), audio_features, pos_embed, cache_dyn, cache_sta]
    for i in range(len(aie_inputs)):
        aie_inputs[i] = torch.from_numpy(aie_inputs[i]).to(device)
    
    mindietorch.set_device(args.device_id)
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(args.decoder_decode_aie_path)
    model.eval().to(device)

    with mindietorch.npu.stream(stream):
        aie_out = model(*aie_inputs)
        stream.synchronize()
    if isinstance(aie_out, tuple):
        aie_out = (x.cpu() for x in aie_out)
    else:
        aie_out = aie_out.cpu()
    compare_onnx_aie_output(onnx_out, aie_out, args.sim_threshold)


def parse_args():
    parser = argparse.ArgumentParser()
    # encoder
    parser.add_argument('--encoder_onnx_path',type=str, default='/tmp/models/encoder.onnx')
    parser.add_argument('--encoder_aie_path', type=str, default='/tmp/models/encoder_compiled.ts')
    # decoder_prefill
    parser.add_argument('--decoder_prefill_onnx_path',type=str, default='/tmp/models/decoder_prefill.onnx')
    parser.add_argument('--decoder_prefill_aie_path', type=str, default='/tmp/models/decoder_prefill_compiled.ts')
    # decoder_decode
    parser.add_argument('--decoder_decode_onnx_path',type=str, default='/tmp/models/decoder_decode.onnx')
    parser.add_argument('--decoder_decode_aie_path', type=str, default='/tmp/models/decoder_decode_compiled.ts')
    parser.add_argument('--sim_threshold', type=float, default=0.99)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--ntokens", type=int, default=100)
    parser.add_argument("--nblocks", type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('=== Compare the outputs of ONNX and AIE ===')

    print('Start comparing encoder...')
    funcs = [compare_encoder, compare_decoder_prefill, compare_decoder_decode]
    for func in funcs:
        func(args)


if __name__ == "__main__":
    main()
