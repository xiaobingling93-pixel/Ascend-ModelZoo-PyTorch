import argparse

import torch
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import mindietorch

from utils import init_encoder_states, build_encoder_input_output


def compare_onnx_aie_output(onnx_out, aie_out, sim_threshold=0.99):
    num_sim = 0
    for i, (a, b) in enumerate(zip(onnx_out, aie_out)):
        a = a.reshape(1, -1).astype(np.float32)
        b = b.cpu().reshape(1, -1)
        sim = F.cosine_similarity(torch.from_numpy(a), b, dim=1)
        if sim > sim_threshold:
            num_sim += 1
        else:
            print(f'Output {i} similarity: {sim}')

    print(f'Number of outputs to compare: {len(onnx_out)}')
    print(f'Number of outputs with cosine similarity > {sim_threshold}: {num_sim}')


def compare_encoder(onnx_path, aie_model_path, sim_threshold=0.99, device_id=0):
    onnx_model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    x = np.ones((1, 45, 80), dtype=np.float32)
    states = init_encoder_states(onnx_model.get_modelmeta().custom_metadata_map, batch_size=1)
    encoder_input, encoder_output_names = build_encoder_input_output(x, states)
    onnx_out = onnx_model.run(encoder_output_names, encoder_input)

    inputs = [x]
    for state in states:
        inputs.append(state)

    mindietorch.set_device(device_id)
    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(aie_model_path)
    model.eval()

    for i in range(len(inputs)):
        inputs[i] = torch.from_numpy(inputs[i]).to(device)

    with mindietorch.npu.stream(stream):
        aie_out = model(*inputs)
        stream.synchronize()

    compare_onnx_aie_output(onnx_out, aie_out, sim_threshold)


def compare_decoder(onnx_path, aie_model_path, sim_threshold=0.99, device_id=0):
    onnx_model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummpy_input = np.ones((1, 2), dtype=np.int32)
    onnx_out = onnx_model.run(
        [onnx_model.get_outputs()[0].name],
        {onnx_model.get_inputs()[0].name: dummpy_input}
    )

    mindietorch.set_device(device_id)
    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)

    model = torch.jit.load(aie_model_path)
    model.eval()
    dummpy_input = torch.from_numpy(dummpy_input).to(device)

    with mindietorch.npu.stream(stream):
        aie_out = model(dummpy_input)
        stream.synchronize()

    compare_onnx_aie_output(onnx_out, [aie_out], sim_threshold)


def compare_joiner(onnx_path, aie_model_path, sim_threshold=0.99, device_id=0):
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

    mindietorch.set_device(device_id)
    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)

    model = torch.jit.load(aie_model_path)
    model.eval()
    encoder_out = torch.from_numpy(encoder_out).to(device)
    decoder_out = torch.from_numpy(decoder_out).to(device)

    with mindietorch.npu.stream(stream):
        aie_out = model(encoder_out, decoder_out)
        stream.synchronize()

    compare_onnx_aie_output(onnx_out, [aie_out], sim_threshold)


def parse_args():
    parser = argparse.ArgumentParser()
    # encoder
    parser.add_argument('--encoder_onnx_path', type=str, help='encoder onnx path')
    parser.add_argument('--encoder_aie_path', type=str, help='encoder aie model path')
    # decoder
    parser.add_argument('--decoder_onnx_path', type=str, help='decoder onnx path')
    parser.add_argument('--decoder_aie_path', type=str, help='decoder aie model path')
    # joiner
    parser.add_argument('--joiner_onnx_path', type=str, help='joiner onnx path')
    parser.add_argument('--joiner_aie_path', type=str, help='joiner aie model path')

    parser.add_argument('--sim_threshold', type=float, default=0.99, help='similarity threshold')
    parser.add_argument('--device_id', type=int, default=0, help='NPU device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('=== Compare the outputs of ONNX and AIE ===')

    print('Start comparing encoder...')
    compare_encoder(args.encoder_onnx_path, args.encoder_aie_path,
                    args.sim_threshold, args.device_id)

    print('Start comparing decoder...')
    compare_decoder(args.decoder_onnx_path, args.decoder_aie_path,
                    args.sim_threshold, args.device_id)

    print('Start comparing joiner...')
    compare_joiner(args.joiner_onnx_path, args.joiner_aie_path,
                   args.sim_threshold, args.device_id)


if __name__ == "__main__":
    main()
