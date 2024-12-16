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

from typing import Tuple, List

import torch
import torch_npu
from tqdm import tqdm
from torch import Tensor

from transformers import AutoTokenizer, T5EncoderModel

from mindiesd.pipeline.pipeline_utils import OpenSoraPipeline
from ..utils import (
    set_random_seed, append_score_to_prompts, extract_prompts_loop,
    merge_prompt, prepare_multi_resolution_info, split_prompt, is_npu_available)
from ..stdit3 import STDiT3
from ..vae import VideoAutoencoder
from ..schedulers import RFlowScheduler

torch_npu.npu.config.allow_internal_format = False
NUM_FRAMES = 'num_frames'


target_image_size = [(720, 1280), (512, 512)]
target_num_frames = [32, 128]
target_fps = [8]
target_output_type = ["latent", "thwc"]
target_dtype = [torch.bfloat16, torch.float16]
MAX_PROMPT_LENGTH = 1024 # the limits of open-sora1.2


class OpenSoraPipeline12(OpenSoraPipeline):

    def __init__(self, text_encoder: T5EncoderModel, tokenizer: AutoTokenizer, transformer: STDiT3,
                 vae: VideoAutoencoder, scheduler: RFlowScheduler,
                 num_frames: int = 32, image_size: Tuple[int, int] = (720, 1280), fps: int = 8,
                 dtype: torch.dtype = torch.bfloat16):

        super().__init__()
        torch.set_grad_enabled(False)

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.num_frames = num_frames
        self.image_size = image_size
        self.fps = fps

        if is_npu_available():
            self.device = 'npu'
        else:
            self.device = 'cpu'

        self.dtype = dtype
        self.text_encoder.to(self.dtype)

    @torch.no_grad()
    def __call__(self, prompts: List[str], seed: int = 42, output_type: str = "latent"):
        stream = torch_npu.npu.Stream(self.device)

        with torch_npu.npu.stream(stream):
            if self.text_encoder.device == torch.device("cpu"):
                self.text_encoder.to(self.device)

        set_random_seed(seed=seed)

        # 1.0 Encode input prompt
        text_encoder_res_list = self._encode_prompt(prompts, self.text_encoder)

        with torch_npu.npu.stream(stream):
            if self.text_encoder.device != torch.device("cpu"):
                self.text_encoder.to('cpu')
        torch.npu.empty_cache()

        input_size = (self.num_frames, *self.image_size)
        latent_size = self.vae.get_latent_size(input_size)

        batch_size = 1
        num_sample = 1

        # == Iter over all samples ==
        all_videos = []
        for i in range(0, len(prompts), batch_size):
            # == prepare batch prompts ==
            batch_prompts = prompts[i: i + batch_size]

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                'STDiT2', (len(batch_prompts), self.image_size, self.num_frames, self.fps), self.device, self.dtype)

            # == Iter over number of sampling for one prompt ==
            for _ in range(num_sample):
                # == Iter over loop generation ==
                z = torch.randn(len(batch_prompts), self.vae.out_channels,
                                *latent_size, device=self.device, dtype=self.dtype)

                # 2.0 Prepare timesteps
                timesteps = self._retrieve_timesteps(z, additional_args=model_args, )

                samples = self._sample(
                    self.transformer,
                    text_encoder_res_list[i],
                    z=z,
                    timesteps=timesteps,
                    additional_args=model_args,
                )

                del z, timesteps, text_encoder_res_list

                samples = self.vae.decode(samples.to(self.dtype), num_frames=self.num_frames)
            all_videos.append(samples)

            del samples
            torch.npu.empty_cache()

        stream.synchronize()

        if not output_type == "latent":
            videos = self._video_write(all_videos)
            return videos
        else:
            return all_videos

    def _video_write(self, x):
        x = [x[0][0]]
        x = torch.cat(x, dim=1)
        value_range = (-1, 1)
        low, high = value_range
        x.clamp_(min=low, max=high)
        x.sub_(low).div_(max(high - low, 1e-5))
        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        return x

    def _retrieve_timesteps(self, z, additional_args=None, ):
        # prepare timesteps
        timesteps = [(1.0 - i / self.scheduler.num_sampling_steps) * self.scheduler.num_timesteps
                     for i in range(self.scheduler.num_sampling_steps)]
        if self.scheduler.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=self.device) for t in timesteps]

        if self.scheduler.use_timestep_transform:
            timesteps = [self._timestep_transform(t, additional_args,
                                                  num_timesteps=self.scheduler.num_timesteps) for t in timesteps]
        return timesteps

    def _sample(self, model, model_args, z, timesteps, additional_args=None, ):

        if additional_args is not None:
            model_args.update(additional_args)

        for i in tqdm(range(0, len(timesteps), 1)):
            t = timesteps[i]
            model_args['t_idx'] = i

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            z = self.scheduler.step(pred, timesteps, i, z)
        return z

    def _timestep_transform(self, t, model_kwargs, num_timesteps):
        base_resolution = 512 * 512
        scale = 1.0

        t = t / num_timesteps
        resolution = model_kwargs["height"].to(torch.float32) * model_kwargs["width"].to(torch.float32)
        ratio_space = (resolution / base_resolution).sqrt()
        # NOTE: currently, we do not take fps into account
        # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
        if model_kwargs[NUM_FRAMES][0] == 1:
            num_frames = torch.ones_like(model_kwargs[NUM_FRAMES])
        else:
            num_frames = model_kwargs[NUM_FRAMES] // 17 * 5
        ratio_time = num_frames.sqrt()

        ratio = ratio_space * ratio_time * scale
        new_t = ratio * t / (1 + (ratio - 1) * t)

        new_t = new_t * num_timesteps
        return new_t

    def _encode(self, text):
        caption_embs, emb_masks = self._get_text_embeddings(text)
        caption_embs = caption_embs[:, None]
        return dict(y=caption_embs, mask=emb_masks)

    def _null(self, n):
        null_y = self.transformer.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y

    def _get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=300,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask

    def _extract_text_res(self, prompts, text_encoder):
        text_encoder_res = self._encode(prompts)
        n = len(prompts)
        y_null = self._null(n)

        text_encoder_res["y"] = torch.cat([text_encoder_res["y"], y_null], 0)
        return text_encoder_res

    def _encode_prompt(self, prompts, text_encoder):
        cfg_aes = 6.5
        cfg_flow = None
        cfg_camera_motion = None
        text_encoder_res_list = []
        for i in range(len(prompts)):
            # == prepare batch prompts ==
            batch_prompts = prompts[i: i + 1]

            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg_aes,
                    flow=cfg_flow,
                    camera_motion=cfg_camera_motion,
                )

            # merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            batch_prompts_loop = extract_prompts_loop(batch_prompts)

            text_encoder_res_list.append(self._extract_text_res(batch_prompts_loop, text_encoder))

        return text_encoder_res_list