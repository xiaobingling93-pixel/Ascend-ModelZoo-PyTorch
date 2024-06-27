import sys
import numpy as np
from ais_bench.infer.interface import InferSession
import torch
import mindietorch
from torch.nn.functional import cosine_similarity


def is_close_to_ones(x1, atol=1e-04):
    x2 = torch.ones_like(x1)
    return torch.allclose(x1, x2, atol)


def precision_test(om_output, onnx_output):
    result = is_close_to_ones(cosine_similarity(om_output, onnx_output))
    print("Precision test passed" if result else "Precision test failed")


def run_infer_session(session, inputs, custom_sizes=None):
    if custom_sizes is not None:
        return session.infer(inputs, 'dymshape', custom_sizes=custom_sizes)
    else:
        return session.infer(inputs)


def run_ts_inference(ts_path, dummpy_input, device_id):
    batch_size = 1
    device = f'npu:{device_id}'
    stream = mindietorch.npu.Stream(device)
    model = torch.jit.load(ts_path)
    model.eval()

    with mindietorch.npu.stream(stream):
        ts_out = model(*dummpy_input)
        stream.synchronize()
    return ts_out


def evaluate_model(mode, om_path, ts_path, device_id):
    session = InferSession(0, om_path)

    if mode == 'encoder':
        x, x_lens = np.random.rand(1, 100, 80).astype(np.float32), np.array([100])
        output_size = 100000
        om_outputs = run_infer_session(session, [x, x_lens], custom_sizes=output_size)

        x_tensor, x_lens_tensor = torch.from_numpy(x), torch.from_numpy(x_lens)
        x_npu_tensor, x_lens_npu_tensor = x_tensor.to(f"npu:{device_id}"), x_lens_tensor.to(f"npu:{device_id}")
        ts_out = run_ts_inference(ts_path, (x_npu_tensor, x_lens_npu_tensor), device_id)

    elif mode == 'decoder':
        y = np.random.randint(0, 10, size=(1, 2)).astype(np.int64)
        om_outputs = run_infer_session(session, [y])

        y_tensor = torch.from_numpy(y)
        y_npu_tensor = y_tensor.to(f'npu:{device_id}')
        ts_out = run_ts_inference(ts_path, (y_npu_tensor,), device_id)

    elif mode == 'joiner':
        enc, dec = np.random.rand(1, 512).astype(np.float32), np.random.rand(1, 512).astype(np.float32)
        om_outputs = run_infer_session(session, [enc, dec])

        enc_tensor, dec_tensor = torch.from_numpy(enc), torch.from_numpy(dec)
        enc_npu_tensor, dec_npu_tensor = enc_tensor.to(f'npu:{device_id}'), dec_tensor.to(f'npu:{device_id}')
        ts_out = run_ts_inference(ts_path, (enc_npu_tensor, dec_npu_tensor), device_id)

    else:
        raise ValueError("Invalid mode")

    om_output = torch.from_numpy(om_outputs[0])
    ts_out = ts_out.to("cpu")
    precision_test(om_output, ts_out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <script> [encoder|decoder|joiner] <om_path> <ts_path>")
        sys.exit(1)

    mode = sys.argv[1]
    om_path = sys.argv[2]
    ts_path = sys.argv[3]
    print("Evaluating precision...")
    device_id = 0
    evaluate_model(mode, om_path, ts_path, device_id)
