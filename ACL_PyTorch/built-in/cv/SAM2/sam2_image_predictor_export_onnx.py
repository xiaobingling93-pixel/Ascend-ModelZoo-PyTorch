# Copyright 2026 Huawei Technologies Co., Ltd
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

import argparse
import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base


class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, input_image: torch.Tensor) -> tuple:
        """
        Args:
          input_image: Input tensor of shape bs x 3 x 1024 x 1024.
        
        Returns:
          high_res_feats_0: High-resolution feature map of shape bs x 32 x 256 x 256,
            used for fine boundary segmentation.
          high_res_feats_1: Medium-resolution feature map of shape bs x 64 x 128 x 128,
            used for medium-level details.
          image_embed: Low-resolution image embedding of shape  bs x 256 x 64 x 64,
            used for semantic understanding in the mask decoder.
        """
        batch_size = input_image.shape[0]
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]
        return feats[0], feats[1], feats[2] # high_res_feats_0, high_res_feats_1, image_embed


class SAM2ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base, multimask_output: bool) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.multimask_output = multimask_output
    
    def forward(self, image_embed: torch.Tensor, high_res_feats_0: torch.Tensor, high_res_feats_1: torch.Tensor,
            point_coords: torch.Tensor, point_labels: torch.Tensor, mask_input: torch.Tensor, has_mask_input: torch.Tensor,
        ):
        """
        Args:
          image_embed: Image embedding tensor of shape 1 x 256 x 64 x 64, output from
            SAM2ImageEncoder.
          high_res_feats_0: High-resolution feature map of shape 1 x 32 x 256 x 256,
            output from SAM2ImageEncoder.
          high_res_feats_1: Medium-resolution feature map of shape 1 x 64 x 128 x 128,
            output from SAM2ImageEncoder.
          point_coords: Point prompt coordinates of shape 1 x N x 2, where N is the
            number of points. Coordinates should be normalized to [0, 1] range.
          point_labels: Point prompt labels of shape 1 x N, where each value is
            1 (foreground), 0 (background), or -1 (ignore).
          mask_input: Low-resolution mask input of shape 1 x 1 x 256 x 256, used for
            iterative refinement. Can be from previous iteration's low_res_masks.
          has_mask_input: Binary flag tensor of shape 1, where 1 indicates
            mask_input is provided, and 0 indicates no mask input.

        Returns:
          masks: The output masks in 1 x 1 x 256 x 256 format, can be resized back 
            to the original input image dimensions using bilinear interpolation.
          iou_predictions: An array of length 1 x 1 containing the model's
            predictions for the quality of each mask.
          low_res_masks: An array of shape 1 x 1 x 256 x 256. These low resolution 
            logits can be passed to a subsequent iteration as mask input.
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        self.sparse_embedding = sparse_embedding
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        high_res_feats = [high_res_feats_0, high_res_feats_1]
        image_embed = image_embed
        
        low_res_masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if self.multimask_output:
            low_res_masks = low_res_masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        else:
            low_res_masks, iou_predictions = self.mask_decoder._dynamic_multimask_via_stability(low_res_masks, iou_predictions)
            
        masks = low_res_masks.clone()

        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)

        return masks, iou_predictions, low_res_masks
        
    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        
        padding_point = torch.zeros((point_coords.shape[0], 1, 2), device=point_coords.device)
        padding_label = -torch.ones((point_labels.shape[0], 1), device=point_labels.device)
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)
        
        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size
        
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)
        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (
                point_labels == -1
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
                1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding


def export_encoder(model: SAM2Base, encoder_output: str, opset: int, batch_size: int):
    sam2_image_encoder = SAM2ImageEncoder(sam_model=model)
    dummy_image_input = torch.randn(batch_size, 3, 1024, 1024, dtype=torch.float)
    print(f"Exporing sam2_image_encoder onnx model to {encoder_output}...")
    torch.onnx.export(
        sam2_image_encoder,
        dummy_image_input,
        encoder_output,
        export_params=True,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['high_res_feats_0', 'high_res_feats_1', 'image_embed']
    )


def export_decoder(model: SAM2Base, decoder_output: str, opset: int, point_nums: int):
    sam2_image_decoder = SAM2ImageDecoder(sam_model=model, multimask_output=False)
    input_image_size = 1024
    point_coords = torch.randint(low=0, high=input_image_size, size=(1, point_nums, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(1, point_nums), dtype=torch.int8)
    mask_input = torch.randn(1, 1, 256, 256, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.int8)
    
    img = torch.randn(1, 3, input_image_size, input_image_size)
    sam2_image_encoder = SAM2ImageEncoder(sam_model=model)
    high_res_feats_0, high_res_feats_1, image_embed = sam2_image_encoder(img)

    masks, iou_predictions, low_res_masks = sam2_image_decoder(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input)
    print(f"Exporing sam2_image_encoder onnx model to {decoder_output}...")
    torch.onnx.export(sam2_image_decoder,
        (image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input),
        decoder_output,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['image_embed', 'high_res_feats_0', 'high_res_feats_1', 'point_coords', 'point_labels', 'mask_input', 'has_mask_input'],
        output_names=['masks', 'iou_predictions', 'low_res_masks']
    )


parser = argparse.ArgumentParser(
    description="Export the SAM2 encoder and decoder to an ONNX model."
)

parser.add_argument("--model_cfg", type=str, required=True, help="Which type of SAM2 model to export.")
parser.add_argument("--checkpoint", type=str, required=True, help="The path to the SAM2 model checkpoint.")
parser.add_argument("--opset", type=int, default=14, help="The ONNX opset version to use.")
parser.add_argument("--encoder_output", type=str, required=True, help="The filename to save the encoder ONNX model to.")
parser.add_argument("--decoder_output", type=str, required=True, help="The filename to save the decoder ONNX model to.",)
parser.add_argument("--bs", type=int, default=1)
parser.add_argument("--pointnums", type=int, default=2)


def run_export(
    model_cfg: str,
    checkpoint: str,
    opset: int,
    encoder_output: str,
    decoder_output: str,
    batch_size: int,
    point_nums: int,
):
    print("Loading model...")
    sam2 = build_sam2(model_cfg, checkpoint, device="cpu")
    print("Loading model end")
    export_encoder(sam2, encoder_output, opset, batch_size)
    export_decoder(sam2, decoder_output, opset, point_nums)

if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        opset=args.opset,
        encoder_output=args.encoder_output,
        decoder_output=args.decoder_output,
        batch_size=args.bs,
        point_nums=args.pointnums
    )