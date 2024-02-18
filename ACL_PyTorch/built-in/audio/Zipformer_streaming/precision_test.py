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
from ais_bench.infer.interface import InferSession

from utils import init_encoder_states, build_encoder_input_output


def compare_onnx_om_output(onnx_out, om_out, sim_threshold=0.99):
    num_sim = 0
    for i, (a, b) in enumerate(zip(onnx_out, om_out)):
        a = a.reshape(1, -1).astype(np.float32)
        b = b.reshape(1, -1)
        sim = F.cosine_similarity(torch.from_numpy(a), torch.from_numpy(b), dim=1)

        if sim > sim_threshold:
            num_sim += 1

    print(f'Number of outputs to compare: {len(onnx_out)}')
    print(f'Number of outputs with cosine similarity > {sim_threshold}: {num_sim}')


def compare_encoder(onnx_path, om_path, sim_threshold=0.99, device_id=0):
    onnx_model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    x = np.ones((1, 45, 80), dtype=np.float32)
    states = init_encoder_states(onnx_model.get_modelmeta().custom_metadata_map, batch_size=1)
    encoder_input, encoder_output_names = build_encoder_input_output(x, states)
    onnx_out = onnx_model.run(encoder_output_names, encoder_input)

    inputs = [x]
    for state in states:
        inputs.append(state)

    encoder_om = InferSession(device_id, om_path)
    om_out = encoder_om.infer(inputs, mode='dymshape', custom_sizes=100000)

    compare_onnx_om_output(onnx_out, om_out, sim_threshold)


def compare_decoder(onnx_path, om_path, sim_threshold=0.99, device_id=0):
    onnx_model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummpy_input = np.ones((1, 2), dtype=np.int64)
    onnx_out = onnx_model.run(
        [onnx_model.get_outputs()[0].name],
        {onnx_model.get_inputs()[0].name: dummpy_input}
    )

    decoder_om = InferSession(device_id, om_path)
    om_out = decoder_om.infer([dummpy_input])

    compare_onnx_om_output(onnx_out, om_out, sim_threshold)


def compare_joiner(onnx_path, om_path, sim_threshold=0.99, device_id=0):
    onnx_model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    encoder_out = np.ones((1, 512), dtype=np.float32)
    decoder_out = np.ones((1, 512), dtype=np.float32)
    onnx_out = onnx_model.run(
        [onnx_model.get_outputs()[0].name],
        {
            onnx_model.get_inputs()[0].name: encoder_out,
            onnx_model.get_inputs()[1].name: decoder_out,
        },
    )

    joiner_om = InferSession(device_id, om_path)
    om_out = joiner_om.infer([encoder_out, decoder_out])

    compare_onnx_om_output(onnx_out, om_out, sim_threshold)


def parse_args():
    parser = argparse.ArgumentParser()
    # encoder
    parser.add_argument('--encoder_onnx_path', type=str, help='encoder onnx path')
    parser.add_argument('--encoder_om_path', type=str, help='encoder om path')
    # decoder
    parser.add_argument('--decoder_onnx_path', type=str, help='decoder onnx path')
    parser.add_argument('--decoder_om_path', type=str, help='decoder om path')
    # joiner
    parser.add_argument('--joiner_onnx_path', type=str, help='joiner onnx path')
    parser.add_argument('--joiner_om_path', type=str, help='joiner om path')

    parser.add_argument('--sim_threshold', type=float, default=0.99, help='similarity threshold')
    parser.add_argument('--device_id', type=int, default=0, help='NPU device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('=== Compare the outputs of ONNX and OM ===')

    print('Start comparing encoder...')
    compare_encoder(args.encoder_onnx_path, args.encoder_om_path, args.sim_threshold, args.device_id)

    print('Start comparing decoder...')
    compare_decoder(args.decoder_onnx_path, args.decoder_om_path, args.sim_threshold, args.device_id)

    print('Start comparing joiner...')
    compare_joiner(args.joiner_onnx_path, args.joiner_om_path, args.sim_threshold, args.device_id)


if __name__ == "__main__":
    main()
