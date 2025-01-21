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

import os
from copy import deepcopy
from typing import List
from collections import deque
import inspect

import torch
import torch.nn as nn
from einops import rearrange
from mindiesd import ConfigMixin

from .model_utils import DiffusionModel
from ..layers.vresnet import ResnetBlock3D, VideoResnetBlock2D
from ..layers.norm import Normalize
from ..layers.conv import VideoConv2d, PlanCausalConv3d
from ..layers.utils import resolve_str_to_obj
from ..layers.wavelet import HaarWaveletTransform3D, InverseHaarWaveletTransform3D
from ..layers.activation import get_activation_fn


class DiagonalGaussianDistribution(object):
    def __init__(
            self,
            parameters,
            deterministic=False,
    ):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device, dtype=self.mean.dtype)

    def sample(self):
        # torch.randn: standard normal distribution
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype=self.mean.dtype)
        return x

    def mode(self):
        return self.mean


class Encoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        energy_flow_hidden_size: int = 64,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        l1_dowmsample_block: str = "Downsample",
        l1_downsample_wavelet: str = "HaarWaveletTransform2D",
        l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
        l2_downsample_wavelet: str = "HaarWaveletTransform3D",
    ) -> None:
        super().__init__()
        self.down1 = self._init_down1(base_channels, norm_type, num_resblocks, l1_dowmsample_block)
        self.energy_flow_hidden_size = energy_flow_hidden_size

        self.down2 = self._init_down2(base_channels, norm_type, num_resblocks, l2_dowmsample_block)

        # Connection
        if l1_dowmsample_block == "Downsample": # Bad code. For temporal usage.
            l1_channels = 12
        else:
            l1_channels = 24

        self.connect_l1 = VideoConv2d(
            l1_channels, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        self.connect_l2 = VideoConv2d(
            24, energy_flow_hidden_size, kernel_size=3, stride=1, padding=1
        )
        # Mid
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 2 + energy_flow_hidden_size,
                out_channels=base_channels * 4,
                norm_type=norm_type,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)
        self.norm_out = Normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = PlanCausalConv3d(
            base_channels * 4, latent_dim * 2, kernel_size=3, stride=1, padding=1
        )
        self.wavelet_transform_in = HaarWaveletTransform3D()
        self.wavelet_transform_l1 = resolve_str_to_obj(l1_downsample_wavelet)()
        self.wavelet_transform_l2 = resolve_str_to_obj(l2_downsample_wavelet)()
        self.nonlinearity = get_activation_fn("silu")

    def forward(self, x):
        coeffs = self.wavelet_transform_in(x)
        l1_coeffs = coeffs[:, :3]
        l1_coeffs = self.wavelet_transform_l1(l1_coeffs)
        l1 = self.connect_l1(l1_coeffs)
        l2_coeffs = self.wavelet_transform_l2(l1_coeffs[:, :3])
        l2 = self.connect_l2(l2_coeffs)

        h = self.down1(coeffs)
        h = torch.concat([h, l1], dim=1)
        h = self.down2(h)
        h = torch.concat([h, l2], dim=1)
        h = self.mid(h)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h, (l1_coeffs, l2_coeffs)

    def _init_down1(self, base_channels, norm_type, num_resblocks, l1_dowmsample_block):
        block = nn.Sequential(
                VideoConv2d(24, base_channels, kernel_size=3, stride=1, padding=1),
                *[
                    VideoResnetBlock2D(
                        in_channels=base_channels,
                        out_channels=base_channels,
                        norm_type=norm_type,
                    )
                    for _ in range(num_resblocks)
                ],
                resolve_str_to_obj(l1_dowmsample_block)(in_channels=base_channels, out_channels=base_channels),
            )
        return block
    
    def _init_down2(self, base_channels, norm_type, num_resblocks, l2_dowmsample_block):
        energy_flow_hidden_size = self.energy_flow_hidden_size
        block = nn.Sequential(
            VideoConv2d(
                base_channels + energy_flow_hidden_size,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 2,
                    out_channels=base_channels * 2,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_dowmsample_block)(base_channels * 2, base_channels * 2),
        )
        return block


class Decoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        t_interpolation: str = "nearest",
        connect_res_layer_num: int = 1,
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
    ) -> None:
        super().__init__()
        self.energy_flow_hidden_size = energy_flow_hidden_size
        self.norm_type = norm_type
        self.conv_in = PlanCausalConv3d(
            latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1)
        
        self.mid = self._init_mid(base_channels, attention_type, energy_flow_hidden_size, use_attention)
    
        self.up2 = self._init_up2(base_channels, num_resblocks, t_interpolation, unsample_type=l2_upsample_block)
        self.up1 = self._init_up1(base_channels, num_resblocks, unsample_type=l1_upsample_block)
        self.layer = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    norm_type=norm_type,
                )
                for i in range(2)
            ],
        )
        # Connection
        if l1_upsample_block == "Upsample": # Bad code. For temporal usage.
            l1_channels = 12
        else:
            l1_channels = 24

        self.connect_l1 = self._init_connect(energy_flow_hidden_size, connect_res_layer_num, l1_channels)
        self.connect_l2 = self._init_connect(energy_flow_hidden_size, connect_res_layer_num, 24)

        # Out
        self.norm_out = Normalize(base_channels, norm_type=norm_type)
        self.conv_out = VideoConv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform3D()
        self.inverse_wavelet_transform_l1 = resolve_str_to_obj(l1_upsample_wavelet)()
        self.inverse_wavelet_transform_l2 = resolve_str_to_obj(l2_upsample_wavelet)()
        self.nonlinearity = get_activation_fn("silu")

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        l2_coeffs = self.connect_l2(h[:, -self.energy_flow_hidden_size :])
        l2 = self.inverse_wavelet_transform_l2(l2_coeffs)
        h = self.up2(h[:, : -self.energy_flow_hidden_size])
        l1_coeffs = h[:, -self.energy_flow_hidden_size :]
        l1_coeffs = self.connect_l1(l1_coeffs)
        l1_coeffs[:, :3] = l1_coeffs[:, :3] + l2
        l1 = self.inverse_wavelet_transform_l1(l1_coeffs)

        h = self.up1(h[:, : -self.energy_flow_hidden_size])

        h = self.layer(h)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        h[:, :3] = h[:, :3] + l1
        dec = self.inverse_wavelet_transform_out(h)
        return dec, (l1_coeffs, l2_coeffs)

    def _init_mid(self, base_channels, attention_type, energy_flow_hidden_size, use_attention):
        norm_type = self.norm_type
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                norm_type=norm_type,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + energy_flow_hidden_size,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type)
            )
        return nn.Sequential(*mid_layers)
    
    def _init_up2(self, base_channels, num_resblocks, t_interpolation, unsample_type):
        norm_type = self.norm_type
        up_block = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(unsample_type)(
                base_channels * 4, base_channels * 4, t_interpolation=t_interpolation
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4 + self.energy_flow_hidden_size,
                norm_type=norm_type,
            ),
        )
        return up_block
    
    def _init_up1(self, base_channels, num_resblocks, unsample_type):
        norm_type = self.norm_type
        up_block = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    norm_type=norm_type,
                )
                for i in range(num_resblocks)
            ],
            resolve_str_to_obj(unsample_type)(in_channels=base_channels * 2, out_channels=base_channels * 2),
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                norm_type=norm_type,
            ),
        )
        return up_block

    def _init_connect(self, energy_flow_hidden_size, connect_res_layer_num, conv_channel):
        norm_type = self.norm_type
        connect = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=energy_flow_hidden_size,
                    out_channels=energy_flow_hidden_size,
                    norm_type=norm_type,
                )
                for _ in range(connect_res_layer_num)
            ],
            VideoConv2d(energy_flow_hidden_size, conv_channel, kernel_size=3, stride=1, padding=1),
        ) 
        return connect


class WFVAEModelConfig(ConfigMixin):
    config_name = "config.json"

    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        encoder_num_resblocks: int = 2,
        encoder_energy_flow_hidden_size: int = 64,
        decoder_num_resblocks: int = 2,
        decoder_energy_flow_hidden_size: int = 128,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        t_interpolation: str = "nearest",
        connect_res_layer_num: int = 1,
        scale: List[float] = None,
        shift: List[float] = None,
        l1_dowmsample_block: str = "Downsample",
        l1_downsample_wavelet: str = "HaarWaveletTransform2D",
        l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
        l2_downsample_wavelet: str = "HaarWaveletTransform3D",
        l1_upsample_block: str = "Upsample",
        l1_upsample_wavelet: str = "InverseHaarWaveletTransform2D",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
        l2_upsample_wavelet: str = "InverseHaarWaveletTransform3D",
    ):
        self._init(locals())
        if not scale:
            self.scale = [0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215, 0.18215]
        if not shift: 
            self.shift = [0, 0, 0, 0, 0, 0, 0, 0]

    def _init(self, value):
        init_signature = inspect.signature(self.__init__)
        parameters = init_signature.parameters
        for param_name, _ in parameters.items():
            if param_name != 'self':
                setattr(self, param_name, value[param_name])  


class WFVAEModel(DiffusionModel):
    config_class = WFVAEModelConfig
    weigths_name = "merged.ckpt"

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        # Module config

        self.use_tiling = False
        # Hardcode for now
        self.t_chunk_enc = 8
        self.t_chunk_dec = 2
        self.t_upsample_times = 2
        
        self.use_quant_layer = False
        self.encoder = Encoder(
            latent_dim=config.latent_dim,
            base_channels=config.base_channels,
            num_resblocks=config.encoder_num_resblocks,
            energy_flow_hidden_size=config.encoder_energy_flow_hidden_size,
            use_attention=config.use_attention,
            norm_type=config.norm_type,
            l1_dowmsample_block=config.l1_dowmsample_block,
            l1_downsample_wavelet=config.l1_downsample_wavelet,
            l2_dowmsample_block=config.l2_dowmsample_block,
            l2_downsample_wavelet=config.l2_downsample_wavelet,
            attention_type=config.attention_type
        )
        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            base_channels=config.base_channels,
            num_resblocks=config.decoder_num_resblocks,
            energy_flow_hidden_size=config.decoder_energy_flow_hidden_size,
            use_attention=config.use_attention,
            norm_type=config.norm_type,
            t_interpolation=config.t_interpolation,
            connect_res_layer_num=config.connect_res_layer_num,
            l1_upsample_block=config.l1_upsample_block,
            l1_upsample_wavelet=config.l1_upsample_wavelet,
            l2_upsample_block=config.l2_upsample_block,
            l2_upsample_wavelet=config.l2_upsample_wavelet
        )

        # Set cache offset for trilinear lossless upsample.
        self._set_cache_offset([self.decoder.up2, self.decoder.connect_l2, self.decoder.conv_in, self.decoder.mid], 1)
        self._set_cache_offset([
            self.decoder.up2[-2:], self.decoder.up1, self.decoder.connect_l1, self.decoder.layer], 
            self.t_upsample_times)

    def encode(self, x):
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)

        if self.use_tiling:
            h = self._tile_encode(x)
            l1, l2 = None, None
        else:
            h, (l1, l2) = self.encoder(x)
            if self.use_quant_layer:
                h = self.quant_conv(h)
            
        posterior = DiagonalGaussianDistribution(h)
        return posterior
    
    def decode(self, z):
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)
        
        if self.use_tiling:
            dec = self._tile_decode(z)
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec, _ = self.decoder(z)
            
        return dec
        
    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling
        self._set_causal_cached(use_tiling)
        
    def disable_tiling(self):
        self.enable_tiling(False)

    def load_weights(self, state_dict):
        with torch.no_grad():
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

    def _empty_causal_cached(self, parent):
        for _, module in parent.named_modules():
            if hasattr(module, 'causal_cached'):
                module.causal_cached = deque()
                
    def _set_causal_cached(self, enable_cached=True):
        for _, module in self.named_modules():
            if hasattr(module, 'enable_cached'):
                module.enable_cached = enable_cached
    
    def _set_cache_offset(self, modules, cache_offset=0):
        for module in modules:
            for submodule in module.modules():
                if hasattr(submodule, 'cache_offset'):
                    submodule.cache_offset = cache_offset

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, 'is_first_chunk'):
                module.is_first_chunk = is_first_chunk

    def _build_chunk_start_end(self, t, decoder_mode=False):
        start_end = [[0, 1]]
        start = 1
        end = start
        while True:
            if start >= t:
                break
            end = min(t, end + (self.t_chunk_dec if decoder_mode else self.t_chunk_enc))
            start_end.append([start, end])
            start = end
        return start_end
    
    def _tile_encode(self, x):
        b, c, t, h, w = x.shape
        
        start_end = self._build_chunk_start_end(t)
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            chunk = x[:, :, start:end, :, :]
            chunk = self.encoder(chunk)[0]
            if self.use_quant_layer:
                chunk = self.quant_conv(chunk)
            result.append(chunk)

        return torch.cat(result, dim=2)
    
    def _tile_decode(self, x):
        b, c, t, h, w = x.shape
        
        start_end = self._build_chunk_start_end(t, decoder_mode=True)
        result = []
        for idx, (start, end) in enumerate(start_end):
            self._set_first_chunk(idx == 0)
            
            if end + 1 < t:
                chunk = x[:, :, start:end + 1, :, :]
            else:
                chunk = x[:, :, start:end, :, :]
                
            if self.use_quant_layer:
                chunk = self.post_quant_conv(chunk)
            chunk = self.decoder(chunk)[0]
            if end + 1 < t:
                chunk = chunk[:, :, :-4]
                result.append(chunk.clone())
            else:
                result.append(chunk.clone())
        
        return torch.cat(result, dim=2)




class WFVAEModelWrapper(nn.Module):
    def __init__(self, model_path, subfolder=None, cache_dir=None, **kwargs):
        super(WFVAEModelWrapper, self).__init__()
        self.vae = WFVAEModel.from_pretrained(model_path, **kwargs)
        self.register_buffer('shift', torch.tensor(self.vae.config.shift)[None, :, None, None, None])
        self.register_buffer('scale', torch.tensor(self.vae.config.scale)[None, :, None, None, None])

    @property
    def dtype(self):
        return self.vae.dtype
    
    @property
    def device(self):
        return self.vae.device

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        return cls(model_path, **kwargs)

    def encode(self, x):
        x = (self.vae.encode(x).sample() - self.shift.to(x.device, dtype=x.dtype)) * \
            self.scale.to(x.device, dtype=x.dtype)
        return x
    
    def decode(self, x):
        x = x / self.scale.to(x.device, dtype=x.dtype) + self.shift.to(x.device, dtype=x.dtype)
        x = self.vae.decode(x)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x


ae_stride_config = {
    'WFVAEModel_D8_4x8x8': [4, 8, 8],
    'WFVAEModel_D16_4x8x8': [4, 8, 8],
    'WFVAEModel_D32_4x8x8': [4, 8, 8],
    'WFVAEModel_D32_8x8x8': [8, 8, 8],
}