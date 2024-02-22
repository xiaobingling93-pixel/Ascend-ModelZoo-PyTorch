# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import argparse

from auto_optimizer import OnnxGraph


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/unet/unet.onnx",
        help="Path of the unet onnx model.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/unet",
        help="Path to save the modified model",
    )
    return parser.parse_args()


def cache_unet(model_path, new_model_path, data):
    model = OnnxGraph.parse(model_path)
    model.add_output(data, dtype='float32', shape=[])
    model.save(new_model_path)
    return


def skip_unet(model_path, new_model_path, data):
    model = OnnxGraph.parse(model_path)
    node = model.get_next_nodes(data)[0]
    batch_size = model.inputs[0].shape[0]
    model.add_input('cache', dtype='float32', shape=[batch_size, 640, 64, 64])
    node.inputs[0] = 'cache'
    model.remove_unused_nodes()
    model.save(new_model_path)
    return


def main(args):
    cache_path = os.path.join(args.save_dir, "unet_cache.onnx")
    skip_path = os.path.join(args.save_dir, "unet_skip.onnx")
    cache_name = '/up_blocks.2/upsamplers.0/conv/Conv_output_0'
    cache_unet(args.model, cache_path, cache_name)
    skip_unet(args.model, skip_path, cache_name)
    return


if __name__ =="__main__":
    main(parse_arguments())
