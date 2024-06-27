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


import time
import argparse
import json

import numpy as np
import torch
import mindietorch
from tqdm import tqdm

def test_encoder(aie_path, device_id = 0):
    batch_size = 1
    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)
    print("Start loading ts module...")
    ts = torch.jit.load(aie_path)
    print("Ts module loaded.")
    ts.eval()
    x, x_lens = np.ones((1, 100, 80), dtype=np.float32), np.array([100])

    inputs = (torch.from_numpy(x).to("npu:0"), torch.from_numpy(x_lens).to("npu:0"))
    print("Start infering...")
    # warmup
    for _ in range(10):
        with mindietorch.npu.stream(stream):
            ts(*inputs)
            stream.synchronize()

    # performance test
    num_infer = 100

    start = time.time()
    for _ in tqdm(range(num_infer)):
        with mindietorch.npu.stream(stream):

            ts(*inputs)
            stream.synchronize()
    end = time.time()

    print(f"Encoder latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"Encoder throughput: {num_infer * batch_size / (end - start):.2f} fps")


def test_decoder(aie_path, device_id):
    batch_size = 1
    dummpy_input = np.ones((batch_size, 2), dtype=np.int64)

    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)
    print("Start loading ts module...")
    model = torch.jit.load(aie_path)
    print("Ts module loaded.")
    model.eval()
    dummpy_input = torch.from_numpy(dummpy_input).to(device)

    # warmup
    for _ in range(10):
        with mindietorch.npu.stream(stream):
            model(dummpy_input)
            stream.synchronize()

    # performance test
    num_infer = 100
    start = time.time()
    for _ in tqdm(range(num_infer)):
        with mindietorch.npu.stream(stream):
            model(dummpy_input)
            stream.synchronize()
    end = time.time()

    print(f"Decoder latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"Decoder throughput: {num_infer * batch_size / (end - start):.2f} fps")


def test_joiner(aie_path, device_id):
    batch_size = 1
    encoder_out = np.ones((batch_size, 512), dtype=np.float32)
    decoder_out = np.ones((batch_size, 512), dtype=np.float32)

    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(aie_path)
    model.eval()
    encoder_out = torch.from_numpy(encoder_out).to(device)
    decoder_out = torch.from_numpy(decoder_out).to(device)

    # warmup
    for _ in range(10):
        with mindietorch.npu.stream(stream):
            out = model(encoder_out, decoder_out)
            stream.synchronize()

    # performance test
    num_infer = 100
    start = time.time()
    for _ in range(num_infer):
        with mindietorch.npu.stream(stream):
            model(encoder_out, decoder_out)
            stream.synchronize()
    end = time.time()

    print(f"Joiner latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"Joiner throughput: {num_infer * batch_size / (end - start):.2f} fps")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_aie_path", type=str, required=True)
    parser.add_argument("--decoder_aie_path", type=str, required=True)
    parser.add_argument("--joiner_aie_path", type=str, required=True)
    parser.add_argument("--device_id", type=int, help="NPU device id", default=0)

    args = parser.parse_args()
    return args


def main():
    mindietorch.set_device(0)
    args = parse_args()


    test_encoder(args.encoder_aie_path, args.device_id)
    test_decoder(args.decoder_aie_path, args.device_id)
    test_joiner(args.joiner_aie_path, args.device_id)


if __name__ == "__main__":
    main()
