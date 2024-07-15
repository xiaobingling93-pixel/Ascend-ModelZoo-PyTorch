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
import torch
import mindietorch

_FRAMES = 3000
_HALF_FRAMES = 1500
_MAX_TOKEN = 224
_KV_NUM = 2


def test(inputs, model, stream, meta=""):
    # warmup
    for _ in range(10):
        with mindietorch.npu.stream(stream):
            model(*inputs)
            stream.synchronize()

    # performance test
    num_infer = 100
    start = time.time()
    for _ in range(num_infer):
        with mindietorch.npu.stream(stream):
            model(*inputs)
            stream.synchronize()
    end = time.time()

    print(f"{meta} latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"{meta} throughput: {num_infer / (end - start):.2f} fps")


def test_encoder(args):
    device = f'npu:{args.device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(args.encoder_aie_path)
    model.eval()

    inputs = [
        torch.ones((1, args.n_mels, _FRAMES), dtype=torch.float32).to(device)
    ]

    test(inputs, model, stream, "Encoder")


def test_decoder_prefill(args):
    device = f'npu:{args.device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(args.decoder_prefill_aie_path)
    model.eval()

    assert args.ntokens <= _MAX_TOKEN, f'ntokens can not exceed {_MAX_TOKEN}'

    inputs = [
        torch.ones((args.beam_size, args.ntokens), dtype=torch.float32).to(device),
        torch.ones((1, _HALF_FRAMES, args.hidden), dtype=torch.float32).to(device),
        torch.ones((args.ntokens, args.hidden), dtype=torch.float32).to(device)
    ]
    
    test(inputs, model, stream, "Decoder prefill")


def test_decoder_decode(args):
    device = f'npu:{args.device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(args.decoder_decode_aie_path)
    model.eval()

    inputs = [
        torch.ones((args.beam_size, 1), dtype=torch.float32).to(device),
        torch.ones((1, _HALF_FRAMES, args.hidden), dtype=torch.float32).to(device),
        torch.ones((args.hidden), dtype=torch.float32).to(device),
        torch.ones((args.nblocks, _KV_NUM, args.beam_size, args.ntokens, args.hidden), dtype=torch.float32).to(device),
        torch.ones((args.nblocks, _KV_NUM, 1, _HALF_FRAMES, args.hidden), dtype=torch.float32).to(device),
    ]

    test(inputs, model, stream, "Decoder decode")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_aie_path",
        type=str, default="/tmp/models/encoder_compiled.ts"
    )
    parser.add_argument(
        "--decoder_prefill_aie_path",
        type=str, default="/tmp/models/decoder_prefill_compiled.ts"
    )
    parser.add_argument(
        "--decoder_decode_aie_path",
        type=str, default="/tmp/models/decoder_decode_compiled.ts"
    )
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--ntokens", type=int, default=100)
    parser.add_argument("--nblocks", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--device_id", type=int, help="NPU device id", default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mindietorch.set_device(args.device_id)

    for func in test_encoder, test_decoder_prefill, test_decoder_decode:
        func(args)


if __name__ == "__main__":
    main()
