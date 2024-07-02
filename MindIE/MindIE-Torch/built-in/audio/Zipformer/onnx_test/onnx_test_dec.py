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
from typing import List

import onnxruntime as ort
import torch


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="/path/to/icefall/results/decoder.txt",
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
        decoder_model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts
        self.init_decoder(decoder_model_filename)

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
            providers=["CUDAExecutionProvider"],
        )

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])

        logging.info(f"context_size: {self.context_size}")
        logging.info(f"vocab_size: {self.vocab_size}")

    def run_decoder(self, decoder_input: torch.Tensor, args) -> torch.Tensor:
        i = 0
        inference_time = []
        while i < args.loop:
            inf_start = time.time()
            _ = self.decoder.run(
                [self.decoder.get_outputs()[0].name],
                {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
            )
            inf_end = time.time()
            inf = inf_end - inf_start
            if i >= args.warm_up:    # use warm_up steps to warmup
                inference_time.append(inf)
            i += 1
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]
        return torch.from_numpy(out), inference_time

def save_tensor_arr_to_file(arr, file_path):
    write_sen = ""
    for l in arr:
        for c in l:
            write_sen += str(c) + " "
        write_sen += "\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(write_sen)


def greedy_search(
    model: OnnxModel,
    args,
) -> List[List[int]]:

    decoder_input = torch.zeros(
        1, 2,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out, inference_time = model.run_decoder(decoder_input, args)
    avg_inf_time = sum(inference_time) / len(inference_time) / 1 * 1000
    print('performance(ms)：', avg_inf_time)
    print("throughput(fps): ", 1000 / avg_inf_time)
    save_tensor_arr_to_file(np.array(decoder_out), args.output_filename)

@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))
    model = OnnxModel(
        decoder_model_filename=args.decoder_model_filename,
    )
    greedy_search(
        model=model,
        args=args,
    )
    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
