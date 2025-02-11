import os

import torch
import torch.nn as nn
from mindiesd.config_utils import ConfigMixin
from mindiesd.models.model_utils import DiffusionModel
from diffusers.models import AutoencoderKL

from ..layer import PatchConv3d, Conv3dAdapter
from ..layer import PatchGroupNorm3d, GroupNorm3dAdapter
from ..layer import rearrange_flatten_t, rearrange_unflatten_t

from .vae_temporal import vae_temporal_sd
from ..utils import Patchify, Depatchify


class VideoAutoencoderConfig(ConfigMixin):
    config_name = 'config.json'

    def __init__(
            self,
            from_pretrained,
            set_patch_parallel=False,
            **kwargs,
    ):
        from_pretrained = os.path.join(from_pretrained, "vae_2d")
        vae_2d = dict(from_pretrained=from_pretrained,
                      subfolder="vae",
                      micro_batch_size=4)
        self.vae_2d = vae_2d
        self.freeze_vae_2d = False
        self.micro_frame_size = 17

        self.shift = (-0.10, 0.34, 0.27, 0.98)
        self.scale = (3.85, 2.32, 2.33, 3.06)

        self.set_patch_parallel = set_patch_parallel

        super().__init__(**kwargs)


class VideoAutoencoderKL(nn.Module):
    def __init__(
            self, from_pretrained, micro_batch_size=None, cache_dir=None, subfolder=None
    ):
        super().__init__()

        path_check = os.path.join(from_pretrained, subfolder)

        self.module = AutoencoderKL.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=True,
            subfolder=subfolder,
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        # x shape is : (B, C, T, H, W)
        x_shape0_b = x.shape[0]
        x = rearrange_flatten_t(x)

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i: i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange_unflatten_t(x, b=x_shape0_b)
        return x

    def decode(self, x, **kwargs):
        # x shape is : (B, C, T, H, W)
        x_shape0_b = x.shape[0]
        x = rearrange_flatten_t(x)

        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i: i + bs]
                x_bs = self.module.decode(x_bs / 0.18215).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange_unflatten_t(x, b=x_shape0_b)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size


class VideoAutoencoder(DiffusionModel):
    config_class = VideoAutoencoderConfig

    weigths_name = 'model.safetensors'

    def __init__(self, config: VideoAutoencoderConfig):
        super().__init__(config=config)

        self.set_patch_parallel = config.set_patch_parallel
        self.spatial_vae = VideoAutoencoderKL(**config.vae_2d)
        self.spatial_vae.to("npu")
        self.temporal_vae = vae_temporal_sd()
        self.temporal_vae.to("npu")

        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size([config.micro_frame_size, None, None])[0]

        if config.freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels

        # normalization parameters
        scale = torch.tensor(config.scale)
        shift = torch.tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

        # Patchify and DePatchify
        if self.set_patch_parallel:
            self.patchify = Patchify()
            self.depatchify = Depatchify()

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [self.micro_frame_size, input_size[1], input_size[2]]
            sub_latent_size = self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] // self.micro_frame_size)
            remain_temporal_size = [input_size[0] % self.micro_frame_size, None, None]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def decode(self, z, num_frames):
        if self.set_patch_parallel:
            for _, module in self.temporal_vae.named_modules():
                if isinstance(module, PatchConv3d) or isinstance(module, PatchGroupNorm3d):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, nn.Conv3d):
                        wrapped_submodule = Conv3dAdapter(submodule, isinstance(module, CausalConv3d))
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, nn.GroupNorm):
                        wrapped_submodule = GroupNorm3dAdapter(submodule)
                        setattr(module, subname, wrapped_submodule)

        z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.set_patch_parallel:
            z_patch = self.patchify(z, dim=-1, is_overlap=True)
            x_z_patch = self.temporal_vae.decode(z_patch, num_frames=num_frames)
            x_z = self.depatchify(x_z_patch, dim=-1, is_overlap=True)
            x_z_patch = self.patchify(x_z, dim=-3, is_overlap=True)
            x_patch = self.spatial_vae.decode(x_z_patch)
            x = self.depatchify(x_patch, dim=-3, is_overlap=True)
        elif self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i: i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z)

        return x