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


import sys
from pruned_transducer_stateless5.onnx_pretrained import OnnxModel

import numpy as np
import torch
import mindietorch

from torch.nn.functional import cosine_similarity

# Initialize the ONNX model globally
onnxmodel = OnnxModel("./exp/encoder-epoch-99-avg-1.onnx", "./exp/decoder-epoch-99-avg-1.onnx",
                      "./exp/joiner-epoch-99-avg-1.onnx")


def is_close_to_ones(x1, atol):
    x2 = torch.ones_like(x1)
    return torch.allclose(x1, x2, atol)


def precision_test(ts_output, onnx_output, atol=1e-02):
    result = is_close_to_ones(cosine_similarity(ts_output, onnx_output), atol)
    print("Precision test" + "passed" if result else "failed")


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


def evaluate_model(mode, ts_path, device_id):
    print(f"Evaluating precision of {mode} model")
    if mode == 'encoder':
        # dummy inputs
        x, x_lens = np.random.rand(1, 100, 80).astype(np.float32), np.array([100])
        x_tensor, x_lens_tensor = torch.from_numpy(x), torch.from_numpy(x_lens)
        x_npu_tensor, x_lens_npu_tensor = x_tensor.to(f"npu:{device_id}"), x_lens_tensor.to(f"npu:{device_id}")

        # gpu/npu inference
        ts_out = run_ts_inference(ts_path, (x_npu_tensor, x_lens_npu_tensor), device_id)
        onnx_output, _ = onnxmodel.run_encoder(x_tensor, x_lens_tensor)

    elif mode == 'decoder':
        y = np.random.randint(0, 10, size=(1, 2)).astype(np.int64)
        y_tensor = torch.from_numpy(y)
        y_npu_tensor = y_tensor.to(f'npu:{device_id}')

        ts_out = run_ts_inference(ts_path, (y_npu_tensor,), device_id)
        onnx_output = onnxmodel.run_decoder(y_tensor)

    elif mode == 'joiner':
        enc, dec = np.random.rand(1, 512).astype(np.float32), np.random.rand(1, 512).astype(np.float32)
        enc_tensor, dec_tensor = torch.from_numpy(enc), torch.from_numpy(dec)
        enc_npu_tensor, dec_npu_tensor = enc_tensor.to(f'npu:{device_id}'), dec_tensor.to(f'npu:{device_id}')

        ts_out = run_ts_inference(ts_path, (enc_npu_tensor, dec_npu_tensor), device_id)
        onnx_output = onnxmodel.run_joiner(enc_tensor, dec_tensor)

    else:
        raise ValueError("Invalid mode")

    ts_out = ts_out.to("cpu")

    precision_test(ts_out, onnx_output, atol=1e-02)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <script> [encoder|decoder|joiner] <ts_path>")
        sys.exit(1)

    mode = sys.argv[1]
    ts_path = sys.argv[2]
    print("Evaluating precision...")
    device_id = 0
    evaluate_model(mode, ts_path, device_id)
