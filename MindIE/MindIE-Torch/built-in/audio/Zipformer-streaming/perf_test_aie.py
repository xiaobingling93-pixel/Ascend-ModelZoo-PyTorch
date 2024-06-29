import time
import argparse
import json

import numpy as np
import torch
import mindietorch

from utils import init_encoder_states, build_encoder_input_output


def test_encoder(encoder_meta_data_path, aie_path, device_id):
    batch_size = 1
    x = np.ones((batch_size, 45, 80), dtype=np.float32)

    with open(encoder_meta_data_path, "r") as f:
        encoder_meta = json.load(f)
    states = init_encoder_states(encoder_meta, batch_size=batch_size)

    inputs = [x]
    for state in states:
        inputs.append(state)

    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(aie_path)
    model.eval()

    for i in range(len(inputs)):
        inputs[i] = torch.from_numpy(inputs[i]).to(device)

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

    print(f"Encoder latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"Encoder throughput: {num_infer * batch_size / (end - start):.2f} fps")


def test_decoder(aie_path, device_id):
    batch_size = 1
    dummpy_input = np.ones((batch_size, 2), dtype=np.int32)

    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(aie_path)
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
    for _ in range(num_infer):
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
    parser.add_argument("--encoder_meta_data_path", type=str, required=True)
    parser.add_argument("--encoder_aie_path", type=str, required=True)
    parser.add_argument("--decoder_aie_path", type=str, required=True)
    parser.add_argument("--joiner_aie_path", type=str, required=True)
    parser.add_argument("--device_id", type=int, help="NPU device id", default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mindietorch.set_device(args.device_id)

    test_encoder(args.encoder_meta_data_path, args.encoder_aie_path, args.device_id)
    test_decoder(args.decoder_aie_path, args.device_id)
    test_joiner(args.joiner_aie_path, args.device_id)


if __name__ == "__main__":
    main()
