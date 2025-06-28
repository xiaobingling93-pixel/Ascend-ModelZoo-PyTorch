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

import unittest
import os
import sys
import torch
import torch_npu

import colossalai
from transformers import T5EncoderModel, T5Tokenizer, T5Config

sys.path.append(os.path.split(sys.path[0])[0])

from opensoraplan import OpenSoraPlanPipeline, CausalVAEModelWrapper, LatteT2V
from opensoraplan import compile_pipe, get_scheduler, set_parallel_manager
from opensoraplan import CacheConfig, OpenSoraPlanDiTCacheManager
from opensoraplan.models.causalvae.modeling_causalvae import CausalVAEModel

SEED = 5464
MASTER_PORT = '42043'
PROMPT = ["A cat playing with a ball"]


class TestOpenSoraPlanPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sp_size = int(os.getenv('SP_SIZE', '1'))
        os.environ['WORLD_SIZE'] = f'{sp_size}'
        if sp_size == 1:
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = MASTER_PORT

        colossalai.launch_from_torch({}, seed=SEED)
        set_parallel_manager(sp_size=sp_size, sp_axis=0)

    def setUp(self):
        torch.manual_seed(SEED)
        torch.npu.manual_seed(SEED)
        torch.npu.manual_seed_all(SEED)
        torch.set_grad_enabled(False)
        self.device_id = 0
        self.device = "npu" if torch.npu.is_available() else "cpu"
        self.num_frames = None
        self.images = None
        self.model_path = None

    def test_pipeline_pndm(self):
        pipeline = self._init("PNDM")
        pipeline = compile_pipe(pipeline)
        result = pipeline(prompt=PROMPT, num_inference_steps=5, guidance_scale=7.5).video
        self.assertEqual(result.shape, torch.Size([1, 17, 256, 256, 3]))

    def test_pipeline_patch_compress_ddpm(self):
        pipeline = self._init("DDPM")
        cache_manager = OpenSoraPlanDiTCacheManager(CacheConfig(1, 3, 1, 2, True))
        # compile pipeline and set the cache_manager and cfg_last_step
        pipeline = compile_pipe(pipeline, cache_manager, 3)
        result = pipeline(prompt=PROMPT, num_inference_steps=5, guidance_scale=7.5).video

        ratio = (
            pipeline.transformer.cache_manager.all_block_num / pipeline.transformer.cache_manager.cal_block_num
        )
        self.assertGreater(ratio, 1.1)
        self.assertEqual(result.shape, torch.Size([1, 17, 256, 256, 3]))

    def _init(self, scheduler_type="PNDM"):
        latent_size = (256 // 8, 256 // 8)
        causal_vae_model = CausalVAEModel(attn_resolutions=[])
        vae = CausalVAEModelWrapper(causal_vae_model, latent_size).eval()
        t5_config = T5Config(
            d_model=4096,
            d_ff=10240,
            num_layers=5,
            num_decoder_layers=5,
            num_heads=64,
            feed_forward_proj="gated-gelu",
            decoder_start_token_id=0,
            dense_act_fn="gelu_new",
            is_gated_act=True,
            model_type="t5",
            output_past=True,
            tie_word_embeddings=False,
        )
        text_encoder = T5EncoderModel(t5_config).eval()
        vocab_file_path = os.path.join(os.path.dirname(__file__), "spiece.model")
        tokenizer = T5Tokenizer(vocab_file_path)
        transformer = LatteT2V(
            activation_fn="gelu-approximate",
            attention_bias=True,
            attention_head_dim=72,
            attention_mode="xformers",
            caption_channels=4096,
            cross_attention_dim=1152,
            in_channels=4,
            norm_elementwise_affine=False,
            norm_eps=1e-6,
            norm_type="ada_norm_single",
            num_embeds_ada_norm=1000,
            num_layers=4,
            out_channels=8,
            patch_size=2,
            sample_size=latent_size,
            video_length=5,
        ).to(self.device).eval()
        schedular = get_scheduler(scheduler_type)

        pipeline = OpenSoraPlanPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=schedular,
            video_length=5,
            image_size=256
        )
        return pipeline


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'])
