# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
import logging
import time
import numpy as np
from typing import Tuple

import onnxruntime as ort
import torch


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="/path/to/icefall/results/encoder.txt",
        help="Path to save the output. ",
    )

    parser.add_argument(
        "--loop",
        type=int,
        default=2000,
        help="loop num for testing speed. ",
    )

    parser.add_argument(
        "--warm_up",
        type=int,
        default=5,
        help="warm up loop num for testing speed. ",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    return parser


class OnnxModel:
    def __init__(
            self,
            encoder_model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.init_encoder(encoder_model_filename)

    def init_encoder(self, encoder_model_filename: str):
        self.encoder = ort.InferenceSession(
            encoder_model_filename,
            sess_options=self.session_opts,
            providers=["CUDAExecutionProvider"],
        )

    def run_encoder(
            self,
            x: torch.Tensor,
            x_lens: torch.Tensor,
            args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        i = 0
        inference_time = []
        while i < args.loop:
            inf_start = time.time()
            _ = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
            },
            )
            inf_end = time.time()
            inf = inf_end - inf_start
            if i >= args.warm_up:    # use warm_up steps to warmup
                inference_time.append(inf)
            i += 1
        out = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1]), inference_time


def save_tensor_arr_to_file(arr, file_path):
    write_sen = ""
    for m in arr:
        for l in m:
            for c in l:
                write_sen += str(c) + " "
            write_sen += "\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(write_sen)

@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))
    model = OnnxModel(
        encoder_model_filename=args.encoder_model_filename,
    )
    features = torch.zeros(1, 100, 80, dtype=torch.float32)
    feature_lengths = torch.tensor([100], dtype=torch.int64)
    encoder_out, encoder_out_lens, inference_time = model.run_encoder(features, feature_lengths, args)
    avg_inf_time = sum(inference_time) / len(inference_time) / 1 * 1000
    print('performance(ms)：', avg_inf_time)
    print("throughput(fps): ", 1000 / avg_inf_time)
    save_tensor_arr_to_file(np.array(encoder_out), args.output_filename)
    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
