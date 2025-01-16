import math
import functools
from typing import List, Optional, Tuple, Union, Literal

import torch
import torch_npu

from .parallel_state import get_rank, get_world_size, all_gather
from .parallel_state import get_dp_world_size, get_dp_rank, get_sp_rank, get_sp_world_size, get_sp_group, get_dp_group


def parallelize_transformer(pipe):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: torch.LongTensor = None,
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        temporal_size = hidden_states.shape[1]
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, get_dp_world_size(), dim=0)[get_dp_rank()]
        n, t, c, h, w = hidden_states.shape
        padding = False
        if h // 2 % get_sp_world_size():
            ori_h = h * 8
            new_h = math.ceil(ori_h / 8 / 2 / 8) * 8 * 8 * 2
            padding = True 
            pad_num = (new_h - ori_h) // 8
            freqs_w = w // 2
            hidden_states = torch.cat([hidden_states, torch.zeros(n, t, c, pad_num, w, device=hidden_states.device, dtype=hidden_states.dtype)], dim=-2)

        hidden_states = torch.chunk(hidden_states, get_dp_world_size(), dim=0)[get_dp_rank()]
        hidden_states = torch.chunk(hidden_states, get_sp_world_size(), dim=-2)[get_sp_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_dp_world_size(), dim=0)[get_dp_rank()]
        if encoder_hidden_states.shape[-2] % get_sp_world_size() == 0:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sp_world_size(), dim=-2)[get_sp_rank()]
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb

            def get_rotary_emb_chunk(freqs):
                dim_thw = freqs.shape[-1]
                freqs = freqs.reshape(temporal_size, -1, dim_thw)
                if padding:
                    freqs = freqs.reshape(temporal_size, -1, freqs_w, dim_thw)
                    freqs = freqs.reshape(temporal_size, -1, dim_thw).contiguous()
                    freqs = torch.cat([freqs, torch.zeros(temporal_size, pad_num // 2 * freqs_w, dim_thw, device=freqs.device, dtype=freqs.dtype)], dim=1)
                freqs = torch.chunk(freqs, get_sp_world_size(), dim=-2)[get_sp_rank()]
                freqs = freqs.reshape(-1, dim_thw)
                return freqs

            freqs_cos = get_rotary_emb_chunk(freqs_cos)
            freqs_sin = get_rotary_emb_chunk(freqs_sin)
            image_rotary_emb = (freqs_cos, freqs_sin)
        
        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            timestep=timestep,
            timestep_cond=timestep_cond,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = all_gather(sample.contiguous(), dim=-2, world_size=get_sp_world_size(), group=get_sp_group())
        if get_dp_world_size() == 2:
            sample = all_gather(sample.contiguous(), world_size=get_dp_world_size(), group=get_dp_group())
        if padding:
            sample = sample[:, :, :, :-pad_num, :]
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward
    
    original_patch_embed_forward = transformer.patch_embed.forward
    
    @functools.wraps(transformer.patch_embed.__class__.forward)
    def new_patch_embed(
        self, text_embeds: torch.Tensor, image_embeds: torch.Tensor
    ):
        text_embeds = all_gather(text_embeds.contiguous(), dim=-2, world_size=get_sp_world_size(), group=get_sp_group())
        image_embeds = all_gather(image_embeds.contiguous(), dim=-2, world_size=get_sp_world_size(), group=get_sp_group())
        batch, num_frames, channels, height, width = image_embeds.shape
        text_len = text_embeds.shape[-2]
        output = original_patch_embed_forward(text_embeds, image_embeds)
        text_embeds = output[:, :text_len, :]
        image_embeds = output[:, text_len:, :].reshape(batch, num_frames, -1, output.shape[-1])

        text_embeds = torch.chunk(text_embeds, get_sp_world_size(), dim=-2)[get_sp_rank()]
        image_embeds = torch.chunk(image_embeds, get_sp_world_size(), dim=-2)[get_sp_rank()]
        image_embeds = image_embeds.reshape(batch, -1, image_embeds.shape[-1])
        res = torch.cat([text_embeds, image_embeds], dim=1)
        return res

    new_patch_embed = new_patch_embed.__get__(transformer.patch_embed)
    transformer.patch_embed.forward = new_patch_embed
