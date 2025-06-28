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

from typing import Optional, List, Union
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CacheConfig:
    start_block_idx: int = 0
    end_block_idx: int = 0
    start_step: int = 0
    step_interval: int = 2
    use_cache: bool = False
    use_cache_encoder: bool = False

    def __post_init__(self):
        if not isinstance(self.start_block_idx, int):
            raise TypeError(f"Expected int for start_block_idx, but got {type(self.start_block_idx).__name__}")
        if not isinstance(self.end_block_idx, int):
            raise TypeError(f"Expected int for end_block_idx, but got {type(self.end_block_idx).__name__}")
        if not isinstance(self.start_step, int):
            raise TypeError(f"Expected int for start_step, but got {type(self.start_step).__name__}")
        if not isinstance(self.step_interval, int):
            raise TypeError(f"Expected int for step_interval, but got {type(self.step_interval).__name__}")
        if not isinstance(self.use_cache, bool):
            raise TypeError(f"Expected bool for use_cache, but got {type(self.use_cache).__name__}")
        if not isinstance(self.use_cache_encoder, bool):
            raise TypeError(f"Expected bool for use_cache_encoder, but got {type(self.use_cache_encoder).__name__}")


class DiTCacheManager:
    def __init__(
        self,
        cache_config: CacheConfig
    ):
        """
        DiTCache plugin for the DiT models. Use this class to enable model cache quickly.
        Args:
            start_block_idx: (`int`)
                The index of the block where chaching starts.
            end_block_idx: (`int`)
                The index of the block where chaching starts.
            start_step: (`int`)
                The index of the DiT denoising step where caching starts.
            step_interval: (`int`)
                Interval of caching steps fot DiT denoising.
            use_cache_encoder: (`bool`)
                Whether the DiT models need to compute encoder_hidden_states.
        """
        self.start_block_idx = cache_config.start_block_idx
        self.end_block_idx = cache_config.end_block_idx
        self.start_step = cache_config.start_step
        self.step_interval = cache_config.step_interval
        self.use_cache = cache_config.use_cache
        self.use_cache_encoder = cache_config.use_cache_encoder

        self.cache_hidden_states = None
        self.cache_encoder_hidden_states = None

        if self.start_block_idx > self.end_block_idx:
            raise ValueError("start_block_idx should not be larger than end_block_idx")

    def __call__(
        self,
        current_step: int,
        block_list: Union[List[nn.ModuleList], List[List[nn.ModuleList]]],
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # If current_step is less than cache start step, execute all blocks sequentially.
        if (current_step < self.start_step) or (not self.use_cache):
            for blocks in zip(*block_list):
                hidden_states, encoder_hidden_states = self._forward_blocks(blocks, hidden_states,
                                                                            encoder_hidden_states, **kwargs)
        # go into cache step interval
        else:
            cache_hidden_states = torch.zeros_like(hidden_states)
            cache_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
            for block_idx, blocks in enumerate(zip(*block_list)):
                # when current_step is exactly on the step_interval, compute and record the cache.
                if current_step % self.step_interval == self.start_step % self.step_interval:
                    # record the tensor before DiT denoising.
                    if block_idx == self.start_block_idx:
                        cache_hidden_states = hidden_states.clone()
                        if self.use_cache_encoder:
                            cache_encoder_hidden_states = encoder_hidden_states.clone()

                    hidden_states, encoder_hidden_states = self._forward_blocks(blocks, hidden_states,
                                                                                encoder_hidden_states, **kwargs)
                    # cache the denoising difference.
                    if block_idx == (self.end_block_idx - 1):
                        self.cache_hidden_states = hidden_states - cache_hidden_states
                        if self.use_cache_encoder:
                            self.cache_encoder_hidden_states = encoder_hidden_states - cache_encoder_hidden_states
                else:
                    # if block_idx is not in the interval using cache, execute all blocks sequentially.
                    if block_idx < self.start_block_idx or block_idx >= self.end_block_idx:
                        hidden_states, encoder_hidden_states = self._forward_blocks(blocks, hidden_states,
                                                                                    encoder_hidden_states, **kwargs)
                    # skip intermediate steps until the end_block_idx, overlay the cached denoising difference.
                    elif block_idx == (self.end_block_idx - 1):
                        hidden_states += self.cache_hidden_states
                        if self.use_cache_encoder:
                            encoder_hidden_states += self.cache_encoder_hidden_states

        return hidden_states

    def _forward_blocks(self, blocks, hidden_states, encoder_hidden_states, **kwargs):
        for block in blocks:
            results = block(hidden_states, encoder_hidden_states=encoder_hidden_states, **kwargs)
            if self.use_cache_encoder:
                hidden_states, encoder_hidden_states = results
            else:
                hidden_states = results

        return hidden_states, encoder_hidden_states
