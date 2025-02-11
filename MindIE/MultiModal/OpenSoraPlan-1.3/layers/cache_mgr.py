#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0  
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn


class CacheConfig():
    def __init__(self, method=None):
        self.method = method


class CacheAgentConfig(CacheConfig):
    """
    The DitCache Config.
    """
    def __init__(self, policy_dir: str):
        """
        Args:
        policy_dir: The file containing the policy.
        """
        super().__init__(method="CacheAgent")
        self.policy_dir = policy_dir


class DitCacheConfig(CacheConfig):
    """
    The DitCache Config.
    """
    def __init__(self, step_start: int, step_interval: int, block_start: int, num_blocks: int):
        """
        Args:
        step_start: The starting step for caching.
        step_interval: The interval at which caching should occur.
        block_start: The starting block index for caching.
        num_blocks: The number of blocks to cache.
        """
        super().__init__(method="DitCache")
        self.step_start = step_start
        self.step_interval = step_interval
        self.block_start = block_start
        self.num_blocks = num_blocks


class CacheManager:
    """
    The CacheManager class is interface to manage the cache algorithm.
    """
    def __init__(
        self,
        config:CacheConfig
    ):  
        """
        Args:
            config: The configuration for the cache algorithm.
        """
        if isinstance(config, CacheConfig):
            self.method = config.method
            self.cache_cls = Cache_cls[self.method](**vars(config))

    def __call__(self, block, time_step, block_idx, hidden_states, *args, **kwargs
    ):  
        """
        Args:
            block: The block in the DiT module.
            time_step: The current time step.
            block_idx: The index of the block.
            hidden_states: The hidden states.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        if not self._use_cache(time_step, block_idx):
            old_hidden_states = hidden_states
            if isinstance(block, list):
                for blk in block:
                    hidden_states = blk(hidden_states, *args, **kwargs)
            else:
                hidden_states = block(hidden_states, *args, **kwargs)
            self._update_cache(hidden_states, old_hidden_states, time_step, block_idx)
        else:
            hidden_states += self._get_cache(time_step, block_idx)
        return hidden_states
    
    def _use_cache(self, time_step, block_idx):
        return self.cache_cls.use_cache(time_step, block_idx)

    def _get_cache(self, time_step, block_idx):
        return self.cache_cls.get_cache(time_step, block_idx)

    def _update_cache(self, hidden_states, old_hidden_states, time_step, block_idx):
        self.cache_cls.update_cache(hidden_states, old_hidden_states, time_step, block_idx)


class CacheAgent():
    def __init__(self, policy_dir, **kwargs):
        self.policy_dir = policy_dir
        self.cache = [None for _ in range(32)]

        self.policy = torch.load(policy_dir)

    def use_cache(self, time_step, block_idx):
        if time_step == 0:
            return False
        else:
            return self.policy[time_step - 1, block_idx]

    def get_cache(self, time_step, block_idx):
        return self.cache[block_idx]
    
    def update_cache(self, hidden_states, old_hidden_states, time_step, block_idx):
        delta = hidden_states - old_hidden_states
        self.cache[block_idx] = delta


class DitCache:
    def __init__(self, step_start, step_interval, block_start, num_blocks, **kwargs):
        self.step_start = step_start
        self.step_interval = step_interval
        self.block_start = block_start
        self.num_blocks = num_blocks
        self.block_end = block_start + num_blocks - 1
        self.cache = None
        self.time_cache = {}

    def use_cache(self, time_step, block_idx):
        if time_step < self.step_start:
            return False
        else:
            diftime = time_step - self.step_start
            if diftime not in self.time_cache:
                self.time_cache[diftime] = diftime % self.step_interval == 0
            if self.time_cache[diftime]:
                return False
            elif block_idx < self.block_start or block_idx > self.block_end:
                return False
            else:
                return True

    def get_cache(self, time_step, block_idx):
        if block_idx == self.block_start:
            return self.cache
        else:
            return 0

    def update_cache(self, hidden_states, old_hidden_states, time_step, block_idx):
        diftime = time_step - self.step_start
        # when (time_step - self.step_start) % self.step_interval == 0:
        if time_step >= self.step_start and self.time_cache[diftime]:
            if block_idx == self.block_start:
                self.cache = old_hidden_states
            elif block_idx == self.block_end:
                self.cache = hidden_states - self.cache


Cache_cls = {
    "CacheAgent" : CacheAgent,
    "DitCache" : DitCache
}
