# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import argparse
import os.path as osp
import time
import math
from typing import List, Optional, Tuple, Union
from numpy.linalg import norm

import torch
from torch import nn
from transformers import AutoModel
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

from jinaBertImplementation.configuration_bert import JinaBertConfig
from jinaBertImplementation.modeling_bert import JinaBertModel
from jinaBertImplementation.modeling_bert import JinaBertEncoder as JinaEncoder


def cosine_similarity(a, b):
    return (a @ b.T) / (norm(a) * norm(b))


def parse_args():
    parser = argparse.ArgumentParser(description="jina-embeddings-v2-base-zh infer")
    parser.add_argument("--model_path", required=True,
                        type=str,
                        help="model path(either local directory or huggingface-Hub)")
    parser.add_argument('--warmup', type=int, default=4, help="Warm up times")
    parser.add_argument('--loop', type=int, default=10, help="loop times")
    args = parser.parse_args()
    return args


def create_model(args):
    if osp.exists(args.model_path):
        AutoModel.register(JinaBertConfig, JinaBertModel)
        config = JinaBertConfig.from_pretrained(args.model_path,
                                                trust_remote_code=True)
        model = JinaBertModel.from_pretrained(args.model_path, config=config,
                                              trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh',
                                           trust_remote_code=True)
    return model.npu()


def rewrite_JinaBertSelfAttention_forward(model):
    """
    优化点1：使用一个Linear(qkv)来代替原有的3个Linear
    优化点2：使用NPU自定义融合算子npu_prompt_flash_attention
             来代替scaled_dot_product_attention（该算子在fx图中包含许多小算子，影响host下发和device执行）
    """
    wq = model.query.weight
    wk = model.key.weight
    wv = model.value.weight
    model.qkv = nn.Linear(wq.shape[0], wq.shape[1] + wk.shape[1] + wv.shape[1])
    model.qkv.weight = nn.Parameter(torch.concat([wq, wk, wv], axis=0), requires_grad=False)
    model.qkv.bias = nn.Parameter(torch.concat([model.query.bias,
                                                model.key.bias,
                                                model.value.bias], axis=0),
                                  requires_grad=False)
    del model.query
    del model.key
    del model.value

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        qkv_layers = model.qkv(hidden_states)
        query_layer, key_layer, value_layer = qkv_layers.split([wq.shape[1], wk.shape[1], wv.shape[1]], dim=2)

        B, S, H = hidden_states.shape
        D = model.attention_head_size
        N = int(H / D)
        new_bias = attention_mask + bias
        attn = torch_npu.npu_prompt_flash_attention(query_layer.contiguous(), key_layer.contiguous(), value_layer.contiguous(),
                                                    pse_shift=new_bias.contiguous(),
                                                    atten_mask=None,
                                                    num_heads=N,
                                                    scale_value=1/math.sqrt(D),
                                                    input_layout="BSH")
        return (attn, )
    model.forward = forward


def rewrite_Linear_forward(model):
    """
    nn.Linear成图后会变成transpose+mm+add，动态小shape输入场景下影响调度性能
    使用npu_linear来提升调度性能
    """
    for m in model.modules():
        if type(m) == nn.Linear:
            class NpuLinear(nn.Linear):
                def forward(self, x):
                    return torch_npu.npu_linear(x, self.weight, self.bias)
            tmp = NpuLinear(m.in_features, m.out_features)
            tmp.weight = m.weight
            tmp.bias = m.bias
            m = tmp


def rewrite_JinaBertGLUMLP_forward(model):
    """
    使用一个chunk(等价于split)来代替原有的两个slice
    一方面split单算子性能相对slice要好，另一方面可以节约host下发时间
    """
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated, non_gated = hidden_states.chunk(2, dim=2)
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
    model.forward = forward


def modify_model(model):
    for layer in model.encoder.layer:
        rewrite_JinaBertSelfAttention_forward(layer.attention.self)
        rewrite_JinaBertGLUMLP_forward(layer.mlp)
        rewrite_Linear_forward(model)

    model.npu().eval().half()


if name == '__main__':
    args = parse_args()

    torch_npu.npu.set_compile_mode(jit_compile=False)

    # 设置torchair参数
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backbend = tng.get_npu_backend(compiler_config=config)

    # 模型创建及torchair处理
    model = create_model(args)
    modify_model(model)
    model.forward = torch.compile(model.forward, dynamic=True,
                                  fullgraph=True, backend=npu_backbend)

    with torch.inference_mode():
        for _ in range(args.warmup):
            embeddings = model.encode(
                [
                    'How is the weather today?',
                    '今天天气怎么样?',
                ], convert_to_tensor=True
            )
        # tensor([[0.7862]])
        print(f'embedding输出结果：{cosine_similarity(embeddings[0].cpu().numpy(), embeddings[1].cpu().numpy())}')
        start = time.time()
        for step in range(args.loop):
            embeddings = model.encode(
                                      ['How is the weather today?',
                                       '今天天气怎么样?',
                                      ], convert_to_tensor=True)
        print(f'E2E time = {(time.time() - start) / args.loop *1000}ms')
