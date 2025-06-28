#!/usr/bin/env python
# coding=utf-8
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

from typing import List, Union
from einops import rearrange, repeat
import torch
import torch.nn as nn
from opensoraplan.models.comm import (
    all_to_all_with_pad,
    get_spatial_pad,
    get_temporal_pad,
)
from opensoraplan.models.parallel_mgr import (
    get_sequence_parallel_group,
    use_sequence_parallel
)
from .dit_cache_common import CacheConfig, DiTCacheManager

SLICE_TEMPORAL_PATTERN = '(b T) S d -> b T S d'
CHANGE_TF_PATTERN = '(b t) f d -> (b f) t d'


class OpenSoraPlanDiTCacheManager(DiTCacheManager):
    def __init__(
            self,
            cache_config: CacheConfig,
    ):
        if not isinstance(cache_config, CacheConfig):
            raise TypeError(f"Expected CacheConfig for cache_config, but got {type(cache_config).__name__}")
        super().__init__(cache_config)
        self.temp_pos_embed = None
        self.all_block_num = 0
        self.cal_block_num = 0
        self.delta_cache = None

    def __call__(
            self,
            current_step: int,
            block_list: Union[List[nn.ModuleList], List[List[nn.ModuleList]]],
            hidden_states: torch.Tensor,
            **kwargs
    ):
        num_blocks = len(block_list[0])
        if self.start_block_idx < 0 or self.start_block_idx > num_blocks:
            raise ValueError("start_block_idx is invalid, out of range [0, num_blocks]")
        if self.end_block_idx < 0 or self.end_block_idx > num_blocks:
            raise ValueError("end_block_idx is invalid, out of range [0, num_blocks]")
        # If current_step is less than cache start step, execute all blocks sequentially.
        if (current_step < self.start_step) or (not self.use_cache):
            self.all_block_num += (num_blocks * 2)
            hidden_states = self._forward_blocks(0, num_blocks, block_list, hidden_states, **kwargs)
        # go into cache step interval
        else:
            self.all_block_num += (num_blocks * 2)
            # infer [0, start_block_idx)
            hidden_states = self._forward_blocks(0, self.start_block_idx, block_list, hidden_states, **kwargs)
            # infer [start_block_idx, end_block_idx)
            hidden_states_before_cache = hidden_states.clone()
            if current_step % self.step_interval == self.start_step % self.step_interval:
                hidden_states = self._forward_blocks(self.start_block_idx, self.end_block_idx, block_list,
                                                     hidden_states, **kwargs)
                self.delta_cache = hidden_states - hidden_states_before_cache
            else:
                if self.delta_cache.shape == hidden_states_before_cache.shape:
                    hidden_states = hidden_states_before_cache + self.delta_cache
                else:
                    hidden_states = self._forward_blocks(self.start_block_idx, self.end_block_idx, block_list,
                                                         hidden_states, **kwargs)
            hidden_states = self._forward_blocks(self.end_block_idx, num_blocks, block_list,
                                                 hidden_states, **kwargs)

        return hidden_states

    def _forward_blocks(self, start_idx, end_idx, block_list, hidden_states, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        encoder_hidden_states_spatial = kwargs.get("encoder_hidden_states_spatial")
        encoder_attention_mask = kwargs.get("encoder_attention_mask")
        timestep_spatial = kwargs.get("timestep_spatial")
        timestep_temp = kwargs.get("timestep_temp")
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
        class_labels = kwargs.get("class_labels")
        input_batch_size = kwargs.get("input_batch_size")
        enable_temporal_attentions = kwargs.get("enable_temporal_attentions")
        t_dim = kwargs.get("t_dim")
        s_dim = kwargs.get("s_dim")
        timestep = kwargs.get("timestep")
        for i, (spatial_block, temp_block) in enumerate(
                zip(block_list[0][start_idx:end_idx], block_list[1][start_idx:end_idx])):
            self.cal_block_num += input_batch_size
            hidden_states = spatial_block(
                hidden_states,
                attention_mask,
                encoder_hidden_states_spatial,
                encoder_attention_mask,
                timestep_spatial,
                cross_attention_kwargs,
                class_labels,
            )

            if enable_temporal_attentions:
                if use_sequence_parallel():
                    hidden_states = rearrange(hidden_states, SLICE_TEMPORAL_PATTERN, T=t_dim,
                                              S=s_dim).contiguous()
                    hidden_states, s_dim, t_dim = self._dynamic_switch(hidden_states, s_dim, t_dim,
                                                                       temporal_to_spatial=True)
                    timestep_temp = repeat(timestep, 'b d -> (b p) d', p=s_dim).contiguous()

                # b c f h w, f = 16 + 4
                hidden_states = rearrange(hidden_states, '(b T) S d -> (b S) T d', b=input_batch_size).contiguous()

                if start_idx + i == 0:
                    hidden_states = hidden_states + self.temp_pos_embed

                hidden_states = temp_block(
                    hidden_states,
                    None,  # attention_mask
                    None,  # encoder_hidden_states
                    None,  # encoder_attention_mask
                    timestep_temp,
                    cross_attention_kwargs,
                    class_labels,
                )

                hidden_states = rearrange(hidden_states, CHANGE_TF_PATTERN,
                                          b=input_batch_size).contiguous()
                if use_sequence_parallel():
                    hidden_states = rearrange(hidden_states, SLICE_TEMPORAL_PATTERN, T=t_dim,
                                              S=s_dim).contiguous()
                    hidden_states, s_dim, t_dim = self._dynamic_switch(hidden_states, s_dim, t_dim,
                                                                       temporal_to_spatial=False)
        return hidden_states

    def _dynamic_switch(self, x, s, t, temporal_to_spatial: bool):
        if temporal_to_spatial:
            scatter_dim, gather_dim = 2, 1
            scatter_pad = get_spatial_pad()
            gather_pad = get_temporal_pad()
        else:
            scatter_dim, gather_dim = 1, 2
            scatter_pad = get_temporal_pad()
            gather_pad = get_spatial_pad()

        x = all_to_all_with_pad(
            x,
            get_sequence_parallel_group(),
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            scatter_pad=scatter_pad,
            gather_pad=gather_pad,
        )
        new_s, new_t = x.shape[2], x.shape[1]
        x = rearrange(x, "b t s d -> (b t) s d")
        return x, new_s, new_t
