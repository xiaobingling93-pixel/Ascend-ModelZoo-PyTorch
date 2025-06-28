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

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND, deprecate
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

from opensoraplan.layers.utils import get_1d_sincos_pos_embed
from opensoraplan.utils.log import logger
from opensoraplan.models.parallel_mgr import (
    get_sequence_parallel_group,
    use_sequence_parallel
)
from opensoraplan.models.comm import (
    all_to_all_with_pad,
    gather_sequence,
    get_spatial_pad,
    get_temporal_pad,
    set_spatial_pad,
    set_temporal_pad,
    split_sequence,
)
from opensoraplan.acceleration.dit_cache_common import CacheConfig
from opensoraplan.acceleration.open_sora_plan_dit_cache import OpenSoraPlanDiTCacheManager

from .latte_modules import PatchEmbed, BasicTransformerBlock, BasicTransformerBlockTemporal, AdaLayerNormSingle, \
    Transformer3DModelOutput, CaptionProjection

ADA_NORM_SINGLE = "ada_norm_single"
SLICE_TEMPORAL_PATTERN = '(b T) S d -> b T S d'
CHANGE_TF_PATTERN = '(b t) f d -> (b f) t d'


@dataclass
class LatteParams:
    hidden_states: torch.Tensor
    timestep: Optional[torch.LongTensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    added_cond_kwargs: Dict[str, torch.Tensor] = None
    enable_temporal_attentions: bool = True
    class_labels: Optional[torch.LongTensor] = None
    cross_attention_kwargs: Dict[str, Any] = None
    attention_mask: Optional[torch.Tensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    use_image_num: int = 0
    return_dict: bool = False


class LatteT2V(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            patch_size_t: int = 1,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
            video_length: int = 17,
            attention_mode: str = 'flash'
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.video_length = video_length

        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels,
        # width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None
        self.cache_manager = OpenSoraPlanDiTCacheManager(CacheConfig())

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = linear_cls(in_channels, inner_dim)
            else:
                self.proj_in = conv_cls(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            if sample_size is None or num_vector_embeds is None:
                logger.error("Transformer2DModel over discrete input must provide sample_size and num_embed")
                raise ValueError

            self.height = sample_size[0]
            self.width = sample_size[1]
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )
        elif self.is_input_patches:
            if sample_size is None:
                logger.error("Transformer2DModel over patched input must provide sample_size")
                raise ValueError

            self.height = sample_size[0]
            self.width = sample_size[1]

            self.patch_size = patch_size
            interpolation_scale = self.config.sample_size[0] // 64  # => 64 (= 512 pixart) has interpolation scale 1
            interpolation_scale = max(interpolation_scale, 1)
            self.pos_embed = PatchEmbed(
                height=sample_size[0],
                width=sample_size[1],
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )

        # 3. Define transformers blocks, spatial attention
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    attention_mode=attention_mode
                )
                for d in range(num_layers)
            ]
        )

        # Define temporal transformers blocks
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlockTemporal(  # one attention
                    inner_dim,
                    num_attention_heads,  # num_attention_heads
                    attention_head_dim,  # attention_head_dim 72
                    dropout=dropout,
                    cross_attention_dim=None,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=False,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    attention_mode=attention_mode
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            if use_linear_projection:
                self.proj_out = linear_cls(inner_dim, in_channels)
            else:
                self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches and norm_type != ADA_NORM_SINGLE:
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
        elif self.is_input_patches and norm_type == ADA_NORM_SINGLE:
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == ADA_NORM_SINGLE:
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = CaptionProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = False

        interpolation_scale = self.config.video_length // 5  # => 5 (= 5 our causalvideovae) has interpolation scale 1
        interpolation_scale = max(interpolation_scale, 1)
        temp_pos_embed = get_1d_sincos_pos_embed(inner_dim, video_length, interpolation_scale=interpolation_scale)
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)

    def forward(
            self,
            latte_params: LatteParams,
            t_idx: torch.Tensor = 0,
    ):
        hidden_states = latte_params.hidden_states
        timestep = latte_params.timestep
        encoder_hidden_states = latte_params.encoder_hidden_states
        added_cond_kwargs = latte_params.added_cond_kwargs
        enable_temporal_attentions = latte_params.enable_temporal_attentions
        class_labels = latte_params.class_labels
        cross_attention_kwargs = latte_params.cross_attention_kwargs
        attention_mask = latte_params.attention_mask
        encoder_attention_mask = latte_params.encoder_attention_mask
        use_image_num = latte_params.use_image_num
        return_dict = latte_params.return_dict

        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num
        hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w').contiguous()
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape: [batch, key_tokens]
        # adds singleton query_tokens dimension:[batch, 1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep, 0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0, discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.to(self.dtype)
        # 1 + 4, 1 -> video condition, 4 -> image condition
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            encoder_attention_mask = repeat(encoder_attention_mask, 'b 1 l -> (b f) 1 l', f=frame).contiguous()
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(encoder_attention_mask_video, 'b 1 l -> b (1 f) l',
                                                  f=frame).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat([encoder_attention_mask_video, encoder_attention_mask_image], dim=1)
            encoder_attention_mask = rearrange(encoder_attention_mask, 'b n l -> (b n) l').contiguous().unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            num_patches = height * width

            hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # alrady add positional embeddings

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                batch_size = input_batch_size
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        t_dim = frame + use_image_num
        s_dim = num_patches
        # shard over the sequence dim if sp is enabled
        if use_sequence_parallel():
            set_temporal_pad(t_dim)
            set_spatial_pad(s_dim)
            hidden_states = rearrange(hidden_states, SLICE_TEMPORAL_PATTERN, T=t_dim, S=s_dim).contiguous()
            hidden_states = split_sequence(hidden_states, get_sequence_parallel_group(), dim=1, pad=get_temporal_pad())
            t_dim = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, 'b T S d -> (b T) S d', T=t_dim, S=s_dim).contiguous()

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states.to(self.dtype))  # 3 120 1152

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                encoder_hidden_states_video = repeat(encoder_hidden_states_video, 'b 1 t d -> b (1 f) t d',
                                                     f=frame).contiguous()
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = torch.cat([encoder_hidden_states_video, encoder_hidden_states_image], dim=1)
                encoder_hidden_states_spatial = rearrange(encoder_hidden_states, 'b f t d -> (b f) t d').contiguous()
            else:
                encoder_hidden_states_spatial = repeat(encoder_hidden_states, 'b t d -> (b f) t d',
                                                       f=t_dim).contiguous()

        # prepare timesteps for spatial and temporal block
        timestep_spatial = repeat(timestep, 'b d -> (b f) d', f=t_dim).contiguous()
        timestep_temp = repeat(timestep, 'b d -> (b p) d', p=num_patches).contiguous()

        if self.training:
            for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks,
                                                                self.temporal_transformer_blocks)):
                if self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        spatial_block,
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states_spatial,
                        encoder_attention_mask,
                        timestep_spatial,
                        cross_attention_kwargs,
                        class_labels,
                        use_reentrant=False,
                    )

                    if enable_temporal_attentions:
                        hidden_states = rearrange(hidden_states,
                                                  '(b f) t d -> (b t) f d',
                                                  b=input_batch_size).contiguous()

                        if use_image_num != 0:  # image-video join training
                            hidden_states_video = hidden_states[:, :frame, ...]
                            hidden_states_image = hidden_states[:, frame:, ...]

                            if i == 0:
                                hidden_states_video = hidden_states_video + self.temp_pos_embed

                            hidden_states_video = torch.utils.checkpoint.checkpoint(
                                temp_block,
                                hidden_states_video,
                                None,  # attention_mask
                                None,  # encoder_hidden_states
                                None,  # encoder_attention_mask
                                timestep_temp,
                                cross_attention_kwargs,
                                class_labels,
                                use_reentrant=False,
                            )

                            hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                            hidden_states = rearrange(hidden_states, CHANGE_TF_PATTERN,
                                                      b=input_batch_size).contiguous()

                        else:
                            if i == 0:
                                hidden_states = hidden_states + self.temp_pos_embed

                            hidden_states = torch.utils.checkpoint.checkpoint(
                                temp_block,
                                hidden_states,
                                None,  # attention_mask
                                None,  # encoder_hidden_states
                                None,  # encoder_attention_mask
                                timestep_temp,
                                cross_attention_kwargs,
                                class_labels,
                                use_reentrant=False,
                            )

                            hidden_states = rearrange(hidden_states, CHANGE_TF_PATTERN,
                                                      b=input_batch_size).contiguous()
        else:
            block_list = [self.transformer_blocks, self.temporal_transformer_blocks]
            self.cache_manager.temp_pos_embed = self.temp_pos_embed
            hidden_states = self.cache_manager(t_idx, block_list, hidden_states,
                                               attention_mask=attention_mask,
                                               encoder_hidden_states_spatial=encoder_hidden_states_spatial,
                                               encoder_attention_mask=encoder_attention_mask,
                                               timestep_spatial=timestep_spatial,
                                               timestep_temp=timestep_temp,
                                               cross_attention_kwargs=cross_attention_kwargs,
                                               class_labels=class_labels,
                                               input_batch_size=input_batch_size,
                                               enable_temporal_attentions=enable_temporal_attentions,
                                               t_dim=t_dim,
                                               s_dim=s_dim,
                                               timestep=timestep)

        if use_sequence_parallel():
            hidden_states = rearrange(hidden_states, "(B T) S C -> B T S C", B=input_batch_size, T=t_dim, S=s_dim)
            hidden_states = gather_sequence(hidden_states, get_sequence_parallel_group(), dim=1, pad=get_temporal_pad())
            t_dim, s_dim = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states = rearrange(hidden_states, "B T S C -> (B T) S C", T=t_dim, S=s_dim)

        if self.is_input_patches:
            if self.config.norm_type != ADA_NORM_SINGLE:
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == ADA_NORM_SINGLE:
                embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame + use_image_num).contiguous()
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            output = rearrange(output, '(b f) c h w -> b c f h w', b=input_batch_size).contiguous()

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)

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

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value


def latte_t2v_8b(**kwargs):
    return LatteT2V(num_layers=56, attention_head_dim=72, num_attention_heads=32, patch_size_t=1, patch_size=2,
                    norm_type=ADA_NORM_SINGLE, caption_channels=4096, cross_attention_dim=2304, sample_size=[64, 64],
                    in_channels=4, out_channels=8, **kwargs)