import os
import torch
import torch_npu
ALGO = int(os.getenv('ALGO'))
if ALGO == 2:
    from mindiesd import attention_forward

MAX_TOKEN = 2147483647


def attention_ATB(query_layer, key_layer, value_layer, self_attention):
    query_layer = query_layer.transpose(1, 2)
    key_layer = key_layer.transpose(1, 2)
    value_layer = value_layer.transpose(1, 2)
    query_layer_list = query_layer.split(1, dim=1)
    key_layer_list = key_layer.split(1, dim=1)
    value_layer_list = value_layer.split(1, dim=1)
    output = []
    for_loop = query_layer.shape[1]
    for i in range(for_loop):
        seqlen = torch.tensor([[query_layer.shape[2]], [key_layer.shape[2]]], dtype=torch.int32)
        intensors = [query_layer_list[i], key_layer_list[i], value_layer_list[i], seqlen]
        out = self_attention.forward(intensors)[0]
        output.append(out)
    out_concat = torch.cat(output, dim=1)
    out = out_concat.transpose(1, 2)
    return out


def attention_FAScore(query_layer, key_layer, value_layer, scale):
    query_layer = query_layer.transpose(1, 2)
    key_layer = key_layer.transpose(1, 2)
    value_layer = value_layer.transpose(1, 2)
    query_layer_list = query_layer.split(1, dim=1)
    key_layer_list = key_layer.split(1, dim=1)
    value_layer_list = value_layer.split(1, dim=1)
    output = []
    for_loop = query_layer.shape[1]
    for i in range(for_loop):
        out = torch_npu.npu_fusion_attention(
            query_layer_list[i],
            key_layer_list[i],
            value_layer_list[i],
            head_num=1,
            input_layout="BNSD",
            scale=scale,
            pre_tockens=MAX_TOKEN,
            next_tockens=MAX_TOKEN
        )[0]
        output.append(out)
    out_concat = torch.cat(output, dim=1)
    out = out_concat.transpose(1, 2)
    return out


def attention_LA(query_layer, key_layer, value_layer, scale):
    # BSND
    query_layer_list = query_layer.split(1, dim=2)
    key_layer_list = key_layer.split(1, dim=2)
    value_layer_list = value_layer.split(1, dim=2)
    output = []
    for_loop = query_layer.shape[2]
    for i in range(for_loop):
        out = attention_forward(
            query_layer_list[i],
            key_layer_list[i],
            value_layer_list[i],
            scale=scale,
            opt_mode="manual",
            op_type="ascend_laser_attention"
        )
        output.append(out)
    out_concat = torch.cat(output, dim=2)
    return out_concat

