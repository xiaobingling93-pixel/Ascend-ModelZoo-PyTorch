# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import time
import argparse
from typing import Tuple, Optional

import torch
from torch import nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

from FlagEmbedding import FlagReranker
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings


def parse_args():
    parser = argparse.ArgumentParser(description="bge-reranker-v2-m3 infer")
    parser.add_argument("--model_path", type=str, default="./",
                        help="model local path (either local directory or huggingface-Hub)")
    parser.add_argument('--warmup', type=int, default=4, help="Warm up times")
    parser.add_argument('--loop', type=int, default=10, help="loop times")
    parser.add_argument("--devices", type=str, default="['npu:0']", help="target devices")
    args = parser.parse_args()
    return args


def create_model(args):
    model = FlagReranker(args.model_path, trust_remote_code=True)
    model.target_devices = eval(args.devices)   # model.target_devices默认=['npu:0', 'npu:1', 'npu:2', 'npu:3']
    return model


class MyXLMRobertaEmbeddings(XLMRobertaEmbeddings):
    """
    重写模型的Embedding层
    修改原本XLMRobertaEmbeddings中的create_position_ids_from_input_ids方法
    将 padding_idx 转换为与 input_ids 相同的设备和张量类型
    """
    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        # 将 padding_idx 转换为与 input_ids 相同的设备和张量类型，避免 FakeTensor
        padding_idx = torch.tensor(padding_idx, device=input_ids.device, dtype=input_ids.dtype)

        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # 这里使用重写后的 self.create_position_ids_from_input_ids 方法
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def rewrite_self_attention_forward(model):
    """
    此处有1个优化点: 使用一个Linear(qkv)来代替原有的3个Linear
    """
    # 新建 Linear(qkv) 并设置权重
    wq = model.query.weight
    wk = model.key.weight
    wv = model.value.weight
    model.qkv = nn.Linear(wq.shape[0], wq.shape[1] + wk.shape[1] + wv.shape[1])
    model.qkv.weight = nn.Parameter(torch.concat([wq, wk, wv], dim=0), requires_grad=False)
    model.qkv.bias = nn.Parameter(torch.concat([
        model.query.bias, model.key.bias, model.value.bias
    ], dim=0), requires_grad=False)
    del model.query
    del model.key
    del model.value

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用新的qkv进行计算
        qkv_layers = model.qkv(hidden_states)
        # 使用chunk得到单独的q, k, v
        chunk_size = wq.shape[1]
        query_layer = qkv_layers[:, :, :chunk_size]
        key_layer = qkv_layers[:, :, chunk_size:chunk_size * 2]
        value_layer = qkv_layers[:, :, chunk_size * 2:]

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = model.transpose_for_scores(query_layer)
        key_layer = model.transpose_for_scores(key_layer)
        value_layer = model.transpose_for_scores(value_layer)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, model.all_head_size)

        outputs = (attn_output,)
        if model.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    model.forward = forward


def modify_model(model):
    xlm_roberta_config = model.model.config
    xlm_roberta_embeddings = model.model.roberta.embeddings
    model.model.roberta.embeddings = MyXLMRobertaEmbeddings(xlm_roberta_config)
    model.model.roberta.embeddings.load_state_dict(xlm_roberta_embeddings.state_dict())
    model.model.roberta.embeddings.eval().half()

    for layer in model.model.roberta.encoder.layer:
        rewrite_self_attention_forward(layer.attention.self)

    return model


if __name__ == '__main__':
    args = parse_args()

    torch_npu.npu.set_compile_mode(jit_compile=False)

    # 设置torchair参数
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)

    # 模型创建及torchair处理
    model = create_model(args)
    model = modify_model(model)
    model.model.eval().half()
    model.model.forward = torch.compile(model.model.forward, dynamic=True, fullgraph=True, backend=npu_backend)

    sentences = [
        ['what is panda?', 'hi'],
        ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
    ]

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model.compute_score(sentences, normalize=True)
            print(f"the similarity of {sentences[0]} is: ", output[0])
            print(f"the similarity of {sentences[1]} is: ", output[1])

        start_time = time.time()
        for _ in range(args.loop):
            output = model.compute_score(sentences, normalize=True)
        print(f'E2E time = {(time.time() - start_time) / args.loop * 1000}ms')