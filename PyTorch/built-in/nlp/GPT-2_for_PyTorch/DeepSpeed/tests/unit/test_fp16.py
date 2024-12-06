# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch_npu
import deepspeed.comm as dist

import deepspeed
import deepspeed_npu
import pytest
from deepspeed.ops.adam import FusedAdam
from unit.common import DistributedTest
from unit.simple_model import (
    SimpleModel,
    SimpleOptimizer,
    random_dataloader,
)
from unit.util import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder

try:
    from apex import amp  # noqa: F401

    _amp_available = True
except ImportError:
    _amp_available = False
amp_available = pytest.mark.skipif(
    not _amp_available, reason="apex/amp is not installed"
)

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestAdamFP32EmptyGrad(DistributedTest):
    world_size = 2

    def test(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.00015}},
            "gradient_clipping": 1.0,
            "fp16": {"enabled": False},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=True)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device,
            dtype=torch.float,
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestAdamwFP16Basic(DistributedTest):
    world_size = 1

    def test(self):
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {"enabled": True},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, optimizer=optimizer
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestAdamFP16ZeroOneCycleCompatibility(DistributedTest):
    world_size = 1

    def test(self, zero_stage, use_cpu_offload):
        if (
            use_cpu_offload
            and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]
        ):
            pytest.skip("cpu-adam is not compatible")

        if use_cpu_offload and DEVICE_NAME != 'Ascend910B':
            pytest.skip("device type is not supported, skip this UT!")

        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.00015}},
            "scheduler": {
                "type": "OneCycle",
                "params": {
                    "cycle_first_step_size": 16000,
                    "cycle_first_stair_count": 8000,
                    "decay_step_size": 16000,
                    "cycle_min_lr": 1e-06,
                    "cycle_max_lr": 3e-05,
                    "decay_lr_rate": 1e-07,
                    "cycle_min_mom": 0.85,
                    "cycle_max_mom": 0.99,
                    "decay_mom_rate": 0.0,
                },
            },
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": zero_stage, "cpu_offload": use_cpu_offload},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestZeroEmptyPartition(DistributedTest):
    world_size = 3

    def test(self, zero_stage, use_cpu_offload):
        if (
            use_cpu_offload
            and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]
        ):
            pytest.skip("cpu-adam is not compatible")

        if use_cpu_offload and DEVICE_NAME != 'Ascend910B':
            pytest.skip("device type is not supported, skip this UT!")

        if zero_stage == 3:
            pytest.skip("skip for now")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {"enabled": True, "initial_scale_power": 8},
            "optimizer": {"type": "Adam", "params": {"lr": 0.00015}},
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload,
                "reduce_bucket_size": 100,
                "allgather_bucket_size": 100,
            },
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)

        # Ensure model has 2 parameters, to cause empty partition with DP=3
        assert len(list(model.parameters())) == 2
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )

        # Now make sure things work..
        data_loader = random_dataloader(
            model=model, total_samples=1, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@amp_available
class TestAmp(DistributedTest):
    world_size = 2

    def test_adam_basic(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "amp": {"enabled": True},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, optimizer=optimizer
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_adam_O2(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.00015}},
            "gradient_clipping": 1.0,
            "amp": {"enabled": True, "opt_level": "O2"},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_adam_O2_empty_grad(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.00015}},
            "gradient_clipping": 1.0,
            "amp": {"enabled": True, "opt_level": "O2"},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZero2ReduceScatterOff(DistributedTest):
    world_size = 2

    def test(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.00015}},
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "allgather_bucket_size": 2000000000,
                "reduce_bucket_size": 200000000,
                "overlap_comm": False,
                "reduce_scatter": False,
            },
            "fp16": {"enabled": True},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("stage", [1, 2, 3])
class TestZeroEmptyGrad(DistributedTest):
    world_size = 1

    def test(self, stage):
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": stage},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, optimizer=optimizer
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
