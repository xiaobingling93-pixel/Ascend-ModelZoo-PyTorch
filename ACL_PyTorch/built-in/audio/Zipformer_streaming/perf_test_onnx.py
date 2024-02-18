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

import onnxruntime as ort
import numpy as np

from utils import init_encoder_states, build_encoder_input_output


def test_encoder(encoder_path, provider):
    onnx_model = ort.InferenceSession(encoder_path, providers=[provider])

    batch_size = 1
    x = np.ones((batch_size, 45, 80), dtype=np.float32)
    states = init_encoder_states(onnx_model.get_modelmeta().custom_metadata_map, batch_size=1)
    encoder_input, encoder_output_names = build_encoder_input_output(x, states)

    # warmup
    for _ in range(10):
        onnx_model.run(encoder_output_names, encoder_input)

    # performance test
    num_infer = 100
    start = time.time()
    for _ in range(num_infer):
        onnx_model.run(encoder_output_names, encoder_input)
    end = time.time()

    print(f"Encoder latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"Encoder throughput: {num_infer * batch_size / (end - start):.2f} fps")


def test_decoder(decoder_path, provider):
    onnx_model = ort.InferenceSession(decoder_path, providers=[provider])

    batch_size = 1
    dummpy_input = np.ones((batch_size, 2), dtype=np.int64)

    # warmup
    for _ in range(10):
        onnx_model.run(
            [onnx_model.get_outputs()[0].name],
            {onnx_model.get_inputs()[0].name: dummpy_input}
        )

    # performance test
    num_infer = 100
    start = time.time()
    for _ in range(num_infer):
        onnx_model.run(
            [onnx_model.get_outputs()[0].name],
            {onnx_model.get_inputs()[0].name: dummpy_input}
        )
    end = time.time()

    print(f"Decoder latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"Decoder throughput: {num_infer * batch_size / (end - start):.2f} fps")


def test_joiner(joiner_path, provider):
    onnx_model = ort.InferenceSession(joiner_path, providers=[provider])

    batch_size = 1
    encoder_out = np.ones((batch_size, 512), dtype=np.float32)
    decoder_out = np.ones((batch_size, 512), dtype=np.float32)

    # warmup
    for _ in range(10):
        onnx_model.run(
            [onnx_model.get_outputs()[0].name],
            {
                onnx_model.get_inputs()[0].name: encoder_out,
                onnx_model.get_inputs()[1].name: decoder_out,
            },
        )

    # performance test
    num_infer = 100
    start = time.time()
    for _ in range(num_infer):
        onnx_model.run(
            [onnx_model.get_outputs()[0].name],
            {
                onnx_model.get_inputs()[0].name: encoder_out,
                onnx_model.get_inputs()[1].name: decoder_out,
            },
        )
    end = time.time()

    print(f"Joiner latency: {(end - start) / num_infer * 1000:.2f} ms")
    print(f"Joiner throughput: {num_infer * batch_size / (end - start):.2f} fps")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--decoder_path", type=str, required=True)
    parser.add_argument("--joiner_path", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.use_gpu:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    test_encoder(args.encoder_path, provider)
    test_decoder(args.decoder_path, provider)
    test_joiner(args.joiner_path, provider)


if __name__ == "__main__":
    main()
