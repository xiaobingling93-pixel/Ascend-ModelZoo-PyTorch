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


import json

import torch


def init_encoder_states(encoder_meta, batch_size):
    num_encoder_layers = encoder_meta["num_encoder_layers"]
    encoder_dims = encoder_meta["encoder_dims"]
    cnn_module_kernels = encoder_meta["cnn_module_kernels"]
    left_context_len = encoder_meta["left_context_len"]
    query_head_dims = encoder_meta["query_head_dims"]
    value_head_dims = encoder_meta["value_head_dims"]
    num_heads = encoder_meta["num_heads"]

    def to_int_list(s):
        return list(map(int, s.split(",")))

    num_encoder_layers = to_int_list(num_encoder_layers)
    encoder_dims = to_int_list(encoder_dims)
    cnn_module_kernels = to_int_list(cnn_module_kernels)
    left_context_len = to_int_list(left_context_len)
    query_head_dims = to_int_list(query_head_dims)
    value_head_dims = to_int_list(value_head_dims)
    num_heads = to_int_list(num_heads)

    num_encoders = len(num_encoder_layers)
    states = []
    for i in range(num_encoders):
        num_layers = num_encoder_layers[i]
        key_dim = query_head_dims[i] * num_heads[i]
        embed_dim = encoder_dims[i]
        nonlin_attn_head_dim = 3 * embed_dim // 4
        value_dim = value_head_dims[i] * num_heads[i]
        conv_left_pad = cnn_module_kernels[i] // 2

        for _ in range(num_layers):
            cached_key = torch.zeros(
                left_context_len[i], batch_size, key_dim
            ).numpy()
            cached_nonlin_attn = torch.zeros(
                1, batch_size, left_context_len[i], nonlin_attn_head_dim
            ).numpy()
            cached_val1 = torch.zeros(
                left_context_len[i], batch_size, value_dim
            ).numpy()
            cached_val2 = torch.zeros(
                left_context_len[i], batch_size, value_dim
            ).numpy()
            cached_conv1 = torch.zeros(batch_size, embed_dim, conv_left_pad).numpy()
            cached_conv2 = torch.zeros(batch_size, embed_dim, conv_left_pad).numpy()
            states += [
                cached_key,
                cached_nonlin_attn,
                cached_val1,
                cached_val2,
                cached_conv1,
                cached_conv2,
            ]

    embed_states = torch.zeros(batch_size, 128, 3, 19).numpy()
    states.append(embed_states)
    processed_lens = torch.zeros(batch_size, dtype=torch.int64).numpy()
    states.append(processed_lens)
    return states


def build_encoder_input_output(
    x,
    states,
):
    encoder_input = {"x": x}
    encoder_output = ["encoder_out"]

    def build_inputs_outputs(tensors, i):
        num_cache_each_layer = 6
        if len(tensors) != num_cache_each_layer:
            raise ValueError(f"Expected {num_cache_each_layer} tensors in layer {i}, got {len(tensors)}")

        # (downsample_left, batch_size, key_dim)
        name = f"cached_key_{i}"
        encoder_input[name] = tensors[0]
        encoder_output.append(f"new_{name}")

        # (1, batch_size, downsample_left, nonlin_attn_head_dim)
        name = f"cached_nonlin_attn_{i}"
        encoder_input[name] = tensors[1]
        encoder_output.append(f"new_{name}")

        # (downsample_left, batch_size, value_dim)
        name = f"cached_val1_{i}"
        encoder_input[name] = tensors[2]
        encoder_output.append(f"new_{name}")

        # (downsample_left, batch_size, value_dim)
        name = f"cached_val2_{i}"
        encoder_input[name] = tensors[3]
        encoder_output.append(f"new_{name}")

        # (batch_size, embed_dim, conv_left_pad)
        name = f"cached_conv1_{i}"
        encoder_input[name] = tensors[4]
        encoder_output.append(f"new_{name}")

        # (batch_size, embed_dim, conv_left_pad)
        name = f"cached_conv2_{i}"
        encoder_input[name] = tensors[5]
        encoder_output.append(f"new_{name}")

    for i in range(len(states[:-2]) // 6):
        build_inputs_outputs(states[i * 6 : (i + 1) * 6], i)

    # (batch_size, channels, left_pad, freq)
    name = "embed_states"
    embed_states = states[-2]
    encoder_input[name] = embed_states
    encoder_output.append(f"new_{name}")

    # (batch_size,)
    name = "processed_lens"
    processed_lens = states[-1]
    encoder_input[name] = processed_lens
    encoder_output.append(f"new_{name}")

    return encoder_input, encoder_output
