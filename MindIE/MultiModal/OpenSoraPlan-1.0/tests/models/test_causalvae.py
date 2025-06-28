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

sys.path.append(os.path.split(sys.path[0])[0])

from opensoraplan.models.causalvae.modeling_causalvae import DiagonalGaussianDistribution, CausalVAEModel

SEED = 5464
MASTER_PORT = '42043'
PROMPT = ["A cat playing with a ball"]


class TestDiagonalGaussianDistribution(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu" if torch.npu.is_available() else "cpu")
        self.parameters = torch.randn([10, 20, 10, 20]).to(self.device)
        self.diagonal_gaussian = DiagonalGaussianDistribution(self.parameters)

    def test_sample(self):
        sample = self.diagonal_gaussian.sample()
        self.assertEqual(sample.shape, self.parameters[:, :10].shape)

    def test_kl(self):
        kl = self.diagonal_gaussian.kl()
        self.assertEqual(kl.shape, torch.Size([10]))

    def test_kl_with_other(self):
        other_parameters = torch.randn([10, 20, 10, 20]).to(self.device)
        other_diagonal_gaussian = DiagonalGaussianDistribution(other_parameters)
        kl = self.diagonal_gaussian.kl(other_diagonal_gaussian)
        self.assertEqual(kl.shape, torch.Size([10]))

    def test_kl_with_deterministic(self):
        other_parameters = torch.randn([10, 20, 10, 20]).to(self.device)
        other_diagonal_gaussian = DiagonalGaussianDistribution(other_parameters, deterministic=True)
        kl = other_diagonal_gaussian.kl()
        self.assertEqual(kl.shape, torch.Size([1]))

    def test_nll(self):
        sample = self.diagonal_gaussian.sample()
        nll = self.diagonal_gaussian.nll(sample)
        self.assertEqual(nll.shape, torch.Size([10]))

    def test_mode(self):
        mode = self.diagonal_gaussian.mode()
        self.assertEqual(mode.shape, self.parameters[:, :10].shape)
        self.assertTrue(torch.allclose(mode, self.diagonal_gaussian.mean, atol=1e-1, rtol=1e-1))


class TestCausalVAEModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu" if torch.npu.is_available() else "cpu")
        self.model = CausalVAEModel(attn_resolutions=(8,)).to(self.device)
        self.path = 'test_checkpoint.pth'
        torch.save(self.model.state_dict(), self.path)

    def test_init(self):
        self.assertIsInstance(self.model, CausalVAEModel)

    def test_decode(self):
        z = torch.randn(1, 4, 4, 8, 8).to(self.device)
        dec = self.model.decode(z)
        self.assertIsInstance(dec, torch.Tensor)

    def test_blend_v(self):
        a = torch.randn(1, 4, 4, 8, 8).to(self.device)
        b = torch.randn(1, 4, 4, 8, 8).to(self.device)
        blend = self.model.blend_v(a, b, 8)
        self.assertIsInstance(blend, torch.Tensor)

    def test_blend_h(self):
        a = torch.randn(1, 4, 4, 8, 8).to(self.device)
        b = torch.randn(1, 4, 4, 8, 8).to(self.device)
        blend = self.model.blend_h(a, b, 8)
        self.assertIsInstance(blend, torch.Tensor)

    def test_tiled_decode2d(self):
        z = torch.randn(1, 4, 4, 8, 8).to(self.device)
        dec = self.model.decode(z)
        self.assertIsInstance(dec, torch.Tensor)

    def test_enable_tiling(self):
        self.model.enable_tiling()
        self.assertTrue(self.model.use_tiling)

    def test_disable_tiling(self):
        self.model.disable_tiling()
        self.assertFalse(self.model.use_tiling)
    
    def test_init_from_ckpt(self):
        new_model = CausalVAEModel(attn_resolutions=(16,))
        new_model.init_from_ckpt(self.path, ['loss'])
        self.assertIsInstance(new_model, CausalVAEModel)

    def tearDown(self):
        os.remove(self.path)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'])
