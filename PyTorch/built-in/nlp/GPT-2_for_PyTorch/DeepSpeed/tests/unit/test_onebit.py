# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch_npu
import deepspeed_npu
import deepspeed.comm as dist
import deepspeed
import pytest
import os
import numpy as np

from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
from deepspeed.ops.op_builder import OpBuilder
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from unit.util import required_minimum_torch_version
from deepspeed.accelerator import get_accelerator

PipeTopo = PipeDataParallelTopology

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


if not required_minimum_torch_version(major_version=1, minor_version=8):
    pytest.skip(
        "NCCL-based 1-bit compression requires torch 1.8 or higher",
        allow_module_level=True,
    )

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version[0] > 4:
    pytest.skip(
        "NCCL-based 1-bit compression is not yet supported w. ROCm 5 until cupy supports ROCm 5",
        allow_module_level=True,
    )


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
class TestOneBitAdamBasic(DistributedTest):
    world_size = 2

    def test(self, dtype):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": (dtype == torch.float16),
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device,
            dtype=dtype,
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
class TestOneBitAdamExpAvgMask(DistributedTest):
    world_size = 2

    def test(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
        mask1 = torch.flatten(mask1)
        optimizer_grouped_parameters = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        model, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters,
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        # Test whether the momentum mask works
        for v in optimizer.state.values():
            if v["exp_avg"].size() == mask1.size():
                assert torch.allclose(
                    v["exp_avg"],
                    v["exp_avg"].mul_(mask1.to(device=v["exp_avg"].device)),
                    atol=1e-07,
                ), f"Momentum mask is not working properly"


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
class TestOneBitAdamCheckpointing(DistributedTest):
    world_size = 2

    def test(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        mask2 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
            mask2[1][col] += 1
        mask1 = torch.flatten(mask1)
        mask2 = torch.flatten(mask2)

        optimizer_grouped_parameters_1 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        optimizer_grouped_parameters_2 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask2,
            },
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        optimizer_grouped_parameters_3 = [
            {"params": [param_optimizer[0][1]], "weight_decay": 0.01},
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        model_1, optimizer_1, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_1,
        )
        data_loader = random_dataloader(
            model=model_1,
            total_samples=10,
            hidden_dim=hidden_dim,
            device=model_1.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_1(batch[0], batch[1])
            model_1.backward(loss)
            model_1.step()
        # Test whether momentum mask still exist after saving checkpoint
        assert optimizer_1.optimizer.adam_freeze_key is True
        mask1 = mask1.to(device=optimizer_1.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(
            optimizer_1.param_groups[0]["exp_avg_mask"], mask1, atol=1e-07
        ), f"Incorrect momentum mask"
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        model_1.save_checkpoint(save_folder, tag=None)
        assert torch.allclose(
            optimizer_1.param_groups[0]["exp_avg_mask"], mask1, atol=1e-07
        ), f"Momentum mask should not change after saving checkpoint"

        model_2, optimizer_2, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_2,
        )
        # Test whether momentum mask stays the same after loading checkpoint
        mask2 = mask2.to(device=optimizer_2.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(
            optimizer_2.param_groups[0]["exp_avg_mask"], mask2, atol=1e-07
        ), f"Incorrect momentum mask"
        model_2.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert torch.allclose(
            optimizer_2.param_groups[0]["exp_avg_mask"], mask2, atol=1e-07
        ), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_2.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"
        assert optimizer_2.optimizer.adam_freeze_key is True

        model_3, optimizer_3, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_3,
        )
        optimizer_3.optimizer.freeze_step = 20
        data_loader = random_dataloader(
            model=model_3,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model_3.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_3(batch[0], batch[1])
            model_3.backward(loss)
            model_3.step()
        assert optimizer_3.optimizer.adam_freeze_key is True
        # Test whether momentum mask stays the same after loading checkpoint
        assert (
            "exp_avg_mask" not in optimizer_3.param_groups[0]
        ), f"Incorrect momentum mask"
        model_3.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert (
            "exp_avg_mask" not in optimizer_3.param_groups[0]
        ), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_3.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"
        assert optimizer_3.optimizer.adam_freeze_key is False

    def test_overflow(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "OneBitAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "freeze_step": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model, total_samples=100, hidden_dim=hidden_dim, device=model.device
        )
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            if dist.get_rank() == 0 and n >= 10:
                loss = loss * 1000000.0
            model.backward(loss)
            dist.barrier()
            model.step()
            dist.barrier()
            model.save_checkpoint(save_folder, tag=None)


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
class TestZeroOneAdamBasic(DistributedTest):
    world_size = 2

    def test(self, dtype):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": (dtype == torch.float16),
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device,
            dtype=dtype,
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
class TestZeroOneAdamExpAvgMask(DistributedTest):
    world_size = 2

    def test(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
        mask1 = torch.flatten(mask1)
        optimizer_grouped_parameters = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        model, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters,
        )
        data_loader = random_dataloader(
            model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device
        )
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        # Test whether the momentum mask works
        for v in optimizer.state.values():
            if v["exp_avg"].size() == mask1.size():
                assert torch.allclose(
                    v["exp_avg"],
                    v["exp_avg"].mul_(mask1.to(device=v["exp_avg"].device)),
                    atol=1e-07,
                ), f"Momentum mask is not working properly"


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
class TestZeroOneAdamCheckpointing(DistributedTest):
    world_size = 2

    def test(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        param_optimizer = list(model.named_parameters())
        mask1 = torch.zeros_like(param_optimizer[0][1].data)
        mask2 = torch.zeros_like(param_optimizer[0][1].data)
        for col in range(mask1.size()[1]):
            mask1[0][col] += 1
            mask2[1][col] += 1
        mask1 = torch.flatten(mask1)
        mask2 = torch.flatten(mask2)

        optimizer_grouped_parameters_1 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask1,
            },
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        optimizer_grouped_parameters_2 = [
            {
                "params": [param_optimizer[0][1]],
                "weight_decay": 0.01,
                "exp_avg_mask": mask2,
            },
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        optimizer_grouped_parameters_3 = [
            {"params": [param_optimizer[0][1]], "weight_decay": 0.01},
            {"params": [param_optimizer[1][1]], "weight_decay": 0.01},
        ]

        model_1, optimizer_1, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_1,
        )
        data_loader = random_dataloader(
            model=model_1,
            total_samples=10,
            hidden_dim=hidden_dim,
            device=model_1.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_1(batch[0], batch[1])
            model_1.backward(loss)
            model_1.step()
        # Test whether momentum mask still exist after saving checkpoint
        mask1 = mask1.to(device=optimizer_1.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(
            optimizer_1.param_groups[0]["exp_avg_mask"], mask1, atol=1e-07
        ), f"Incorrect momentum mask"
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        model_1.save_checkpoint(save_folder, tag=None)
        assert torch.allclose(
            optimizer_1.param_groups[0]["exp_avg_mask"], mask1, atol=1e-07
        ), f"Momentum mask should not change after saving checkpoint"

        model_2, optimizer_2, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_2,
        )
        # Test whether momentum mask stays the same after loading checkpoint
        mask2 = mask2.to(device=optimizer_2.param_groups[0]["exp_avg_mask"].device)
        assert torch.allclose(
            optimizer_2.param_groups[0]["exp_avg_mask"], mask2, atol=1e-07
        ), f"Incorrect momentum mask"
        model_2.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert torch.allclose(
            optimizer_2.param_groups[0]["exp_avg_mask"], mask2, atol=1e-07
        ), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_2.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"

        model_3, optimizer_3, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=optimizer_grouped_parameters_3,
        )
        optimizer_3.optimizer.freeze_step = 20
        data_loader = random_dataloader(
            model=model_3,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model_3.device,
        )
        for n, batch in enumerate(data_loader):
            loss = model_3(batch[0], batch[1])
            model_3.backward(loss)
            model_3.step()
        # Test whether momentum mask stays the same after loading checkpoint
        assert (
            "exp_avg_mask" not in optimizer_3.param_groups[0]
        ), f"Incorrect momentum mask"
        model_3.load_checkpoint(
            save_folder,
            tag=None,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        assert (
            "exp_avg_mask" not in optimizer_3.param_groups[0]
        ), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is reset
        for v in optimizer_3.state.values():
            assert "worker_error" not in v, f"Incorrect worker error"
            assert "server_error" not in v, f"Incorrect server error"

    def test_overflow(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "ZeroOneAdam",
                "params": {
                    "lr": 0.00015,
                    "weight_decay": 0.01,
                    "var_freeze_step": 4,
                    "var_update_scaler": 1,
                    "local_step_scaler": 1,
                    "local_step_clipper": 2,
                    "cuda_aware": False,
                    "comm_backend_name": get_accelerator().communication_backend_name(),
                },
            },
            "gradient_clipping": 1.0,
            "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 16},
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model, total_samples=100, hidden_dim=hidden_dim, device=model.device
        )
        save_folder = os.path.join(tmpdir, "saved_checkpoint")
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            if dist.get_rank() == 0 and n >= 10:
                loss = loss * 1000000.0
            model.backward(loss)
            dist.barrier()
            model.step()
            dist.barrier()
            model.save_checkpoint(save_folder, tag=None)
