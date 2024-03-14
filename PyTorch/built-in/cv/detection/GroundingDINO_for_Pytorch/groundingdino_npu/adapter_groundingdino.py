# Copyright 2023 Huawei Technologies Co., Ltd
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

from typing import Optional, no_type_check, Tuple, Union

import torch
import mmengine
from torch import Tensor, nn
from mmengine.model import ModuleList
from mmengine.utils import deprecated_api_warning
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch, MultiScaleDeformableAttnFunction

from mmdet.models.detectors.deformable_detr import DeformableDETR
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoder
import ads.common


@no_type_check
@deprecated_api_warning({'residual': 'identity'},
                        cls_name='MultiScaleDeformableAttention')
def msda_forward(self,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            identity: Optional[torch.Tensor] = None,
            query_pos: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            reference_points: Optional[torch.Tensor] = None,
            spatial_shapes: Optional[torch.Tensor] = None,
            level_start_index: Optional[torch.Tensor] = None,
            **kwargs) -> torch.Tensor:
    """Forward Function of MultiScaleDeformAttention.

    Args:
        query (torch.Tensor): Query of Transformer with shape
            (num_query, bs, embed_dims).
        key (torch.Tensor): The key tensor with shape
            `(num_key, bs, embed_dims)`.
        value (torch.Tensor): The value tensor with shape
            `(num_key, bs, embed_dims)`.
        identity (torch.Tensor): The tensor used for addition, with the
            same shape as `query`. Default None. If None,
            `query` will be used.
        query_pos (torch.Tensor): The positional encoding for `query`.
            Default: None.
        key_padding_mask (torch.Tensor): ByteTensor for `query`, with
            shape [bs, num_key].
        reference_points (torch.Tensor):  The normalized reference
            points with shape (bs, num_query, num_levels, 2),
            all elements is range in [0, 1], top-left (0,0),
            bottom-right (1, 1), including padding area.
            or (N, Length_{query}, num_levels, 4), add
            additional two dimensions is (w, h) to
            form reference boxes.
        spatial_shapes (torch.Tensor): Spatial shape of features in
            different levels. With shape (num_levels, 2),
            last dimension represents (h, w).
        level_start_index (torch.Tensor): The start index of each level.
            A tensor has shape ``(num_levels, )`` and can be represented
            as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

    Returns:
        torch.Tensor: forwarded results with shape
        [num_query, bs, embed_dims].
    """

    if value is None:
        value = query

    if identity is None:
        identity = query
    if query_pos is not None:
        query = query + query_pos

    if not self.batch_first:
        # change to (bs, num_query ,embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

    bs, num_query, _ = query.shape
    bs, num_value, _ = value.shape
    assert (spatial_shapes[:, 0] * spatial_shapes[:,
                                   1]).sum() == num_value

    value = self.value_proj(value)
    if key_padding_mask is not None:
        value = value.masked_fill(key_padding_mask[..., None], 0.0)
    value = value.view(bs, num_value, self.num_heads, -1)
    sampling_offsets = self.sampling_offsets(query).view(
        bs, num_query, self.num_heads, self.num_levels, self.num_points,
        2)
    attention_weights = self.attention_weights(query).view(
        bs, num_query, self.num_heads,
        self.num_levels * self.num_points)
    attention_weights = attention_weights.softmax(-1)

    attention_weights = attention_weights.view(bs, num_query,
                                               self.num_heads,
                                               self.num_levels,
                                               self.num_points)

    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                             + sampling_offsets \
                             / offset_normalizer[None, None, None, :,
                               None, :]
    elif reference_points.shape[-1] == 4:
        sampling_locations = reference_points[:, :, None, :, None, :2] \
                             + sampling_offsets / self.num_points \
                             * reference_points[:, :, None, :, None, 2:] \
                             * 0.5
    else:
        raise ValueError(
            f'Last dim of reference_points must be'
            f' 2 or 4, but get {reference_points.shape[-1]} instead.')

    output = ads.common.npu_multi_scale_deformable_attn_function(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights)

    output = self.output_proj(output)

    if not self.batch_first:
        # (num_query, bs ,embed_dims)
        output = output.permute(1, 0, 2)

    return self.dropout(output) + identity


def gen_encoder_output_proposals(
        self, memory: Tensor, memory_mask: Tensor,
        spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
    """Generate proposals from encoded memory. The function will only be
    used when `as_two_stage` is `True`.

    Args:
        memory (Tensor): The output embeddings of the Transformer encoder,
            has shape (bs, num_feat_points, dim).
        memory_mask (Tensor): ByteTensor, the padding mask of the memory,
            has shape (bs, num_feat_points).
        spatial_shapes (Tensor): Spatial shapes of features in all levels,
            has shape (num_levels, 2), last dimension represents (h, w).

    Returns:
        tuple: A tuple of transformed memory and proposals.

        - output_memory (Tensor): The transformed memory for obtaining
          top-k proposals, has shape (bs, num_feat_points, dim).
        - output_proposals (Tensor): The inverse-normalized proposal, has
          shape (batch_size, num_keys, 4) with the last dimension arranged
          as (cx, cy, w, h).
    """

    bs = memory.size(0)
    proposals = []
    _cur = 0  # start index in the sequence of the current level
    for lvl, HW in enumerate(spatial_shapes):
        H, W = HW

        if memory_mask is not None:
            mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                bs, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                1).unsqueeze(-1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                1).unsqueeze(-1)
            scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
        else:
            if not isinstance(HW, torch.Tensor):
                HW = memory.new_tensor(HW)
            scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(
                0, H, 1, dtype=torch.float32, device=memory.device),
            torch.arange(
                0, W, 1, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
        proposals.append(proposal)
        _cur += (H * W)
    output_proposals = torch.cat(proposals, 1)
    # do not use `all` to make it exportable to onnx
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
            -1, keepdim=True) == output_proposals.shape[-1]
    # inverse_sigmoid
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    if memory_mask is not None:
        output_proposals = output_proposals.masked_fill(
            memory_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(
        ~output_proposals_valid, float('inf'))

    output_memory = memory
    if memory_mask is not None:
        output_memory = output_memory.masked_fill(
            memory_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid,
                                              float(0))
    output_memory = self.memory_trans_fc(output_memory)
    output_memory = self.memory_trans_norm(output_memory)
    # [bs, sum(hw), 2]
    return output_memory, output_proposals


@staticmethod
def get_encoder_reference_points(
        spatial_shapes: Tensor, valid_ratios: Tensor,
        device: Union[torch.device, str]) -> Tensor:
    """Get the reference points used in encoder.

    Args:
        spatial_shapes (Tensor): Spatial shapes of features in all levels,
            has shape (num_levels, 2), last dimension represents (h, w).
        valid_ratios (Tensor): The ratios of the valid width and the valid
            height relative to the width and the height of features in all
            levels, has shape (bs, num_levels, 2).
        device (obj:`device` or str): The device acquired by the
            `reference_points`.

    Returns:
        Tensor: Reference points used in decoder, has shape (bs, length,
        num_levels, 2).
    """

    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(
                0.5, H - 0.5 + 1, 1, dtype=torch.float32, device=device),
            torch.arange(
                0.5, W - 0.5 + 1, 1, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    # [bs, sum(hw), num_level, 2]
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


MultiScaleDeformableAttention.forward = msda_forward
DeformableDETR.gen_encoder_output_proposals = gen_encoder_output_proposals
DeformableDetrTransformerEncoder.get_encoder_reference_points = get_encoder_reference_points
