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
# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from functools import wraps

import torch
import torch.distributed as dist
from einops import rearrange
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel

import opensora
from opensora.utils.config_utils import parse_configs
from opensora.acceleration.parallel_states import (get_sequence_parallel_group,
                                                   get_sequence_parallel_group_for_send_recv_overlap)
from opensora.utils.device_utils import is_npu_available

if not is_npu_available():
    import xformers.ops
else:
    import torch_npu

cfg = parse_configs(training=True)


def attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if cfg.sp_size > 1 and cfg.context_parallel_algo == 'ulysses_cp_algo':
            sp_group = get_sequence_parallel_group()
            if self.enable_flashattn:
                # q.shape: s b h d
                self.core_attention = UlyssesContextAttention(self.core_attention, sp_group)
            else:
                # q.shape: b h s d
                self.core_attention = UlyssesContextAttention(self.core_attention, sp_group,
                                                              scatter_idx=1, gather_idx=2)
    return wrapper


def core_attention_forward(self, q, k, v):
    if not self.enable_flashattn:
        # q.shape: b h s d
        dtype = q.dtype
        q = q * self.scale
        # translate attn to float32
        attn = q @ k.transpose(-2, -1)
        attn = attn.to(torch.float32)
        attn = attn.softmax(dim=-1)
        # cast back attn to original dtype
        attn = attn.to(dtype)
        attn = self.attn_drop(attn)
        x = attn @ v
        return x
    if is_npu_available() and q.dtype in [torch.float16, torch.bfloat16]:
        if cfg.sp_size > 1 and cfg.context_parallel_algo == 'megatron_cp_algo':

            cp_group = get_sequence_parallel_group()
            cp_size = dist.get_world_size(cp_group)
            rank = dist.get_rank(group=cp_group)
            # The following only applies to enabling dp and sp
            local_ranks_in_sp_group = list(range(cp_size))
            cp_global_ranks = []
            current_global_rank = dist.get_rank()
            for local_rank in local_ranks_in_sp_group:
                global_rank = (current_global_rank // cp_size) * cp_size + local_rank
                cp_global_ranks.append(global_rank)

            cp_para = dict()
            cp_para['causal'] = None
            cp_para['cp_group'] = cp_group
            cp_para['cp_size'] = cp_size
            cp_para['rank'] = rank
            cp_para['cp_global_ranks'] = cp_global_ranks
            cp_para['use_cp_send_recv_overlap'] = cfg.use_cp_send_recv_overlap
            cp_para['cp_group_for_send_recv_overlap'] = get_sequence_parallel_group_for_send_recv_overlap() \
                if cfg.use_cp_send_recv_overlap else None
            head_num = q.shape[-2]
            q, k, v = [rearrange(x, 's b h d -> s b (h d)') for x in [q, k, v]]
            x = ringattn_context_parallel(q, k, v, head_num, cp_para, self.scale, None, self.attn_drop.p)
        else:
            num_head = q.shape[-2]
            q, k, v = [rearrange(x, 's b h d -> s b (h d)') for x in [q, k, v]]
            x = torch_npu.npu_fusion_attention(
                q, k, v, num_head, input_layout="SBH",
                pse=None,
                scale=self.scale,
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1. - self.attn_drop.p if self.training else 1.,
                sync=False,
                inner_precise=0,
            )[0]
    else:
        from flash_attn import flash_attn_func

        x = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.scale,
        )
    return x

def exe_adaptation():
    opensora.models.layers.blocks.AttentionWithCp.__init__ = attention_init_wrapper(
        opensora.models.layers.blocks.AttentionWithCp.__init__)
    opensora.models.layers.blocks.CoreAttention.forward = core_attention_forward


exe_adaptation()