# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from collections import namedtuple
from typing import Dict, List, NamedTuple, Set, Tuple
import pytest
import deepspeed.comm as dist
import torch
import torch_npu
import deepspeed_npu
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.loss import L1Loss
from torch.nn.parameter import Parameter

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.utils import ZeRORuntimeException
from deepspeed.accelerator import get_accelerator

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def run_unbalanced_gradients(model, data_loader):

    def drop_some_gradients(model, iter):
        odd_iteration = iter % 2
        for i, p in enumerate(model.parameters()):
            p.requires_grad = (i % 2) == odd_iteration

    def enable_grads(model):
        for p in model.parameters():
            p.requires_grad = True

    for i, batch in enumerate(data_loader):
        drop_some_gradients(model, i + 1)
        loss = model(batch[0], batch[1])
        model.backward(loss)
        model.step()
        enable_grads(model)


def dump_state_dict(model):
    if dist.get_rank() == 0:
        print("state_dict:")
        for name, param in model.named_parameters():
            print(f"{name} {param.data}")


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUnbalancedGradients(DistributedTest):
    world_size = 1

    def test(self, zero_stage):
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {"stage": zero_stage},
            "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
            "fp16": {"enabled": True, "initial_scale_power": 8},
        }
        hidden_dim = 4

        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict, model=model, model_parameters=model.parameters()
        )
        data_loader = random_dataloader(
            model=model, total_samples=16, hidden_dim=hidden_dim, device=model.device
        )

        run_unbalanced_gradients(model, data_loader)


def _ds_initialize_for_param_partitioning_testing(
    model: Module, cfg: dict
) -> DeepSpeedEngine:
    ds_engine, _, _, _ = deepspeed.initialize(
        config=cfg, model=model, model_parameters=model.parameters()
    )

    return ds_engine


def _assert_partition_status(
    model: Module, valid_statuses: Set[ZeroParamStatus]
) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status in valid_statuses, param.ds_summary()


def _assert_fully_available(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_status == ZeroParamStatus.AVAILABLE


class EltwiseMultiplicationModule(Module):

    def __init__(self, weight: Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        _assert_fully_available(self)
        result = self.weight * x

        return result


class EltwiseMultiplicationTestNetwork_Dict(Module):
    """used for testing purposes"""

    def __init__(
        self,
        weight1: Parameter,
        weight2: Parameter,
        weight3: Parameter,
    ) -> None:
        super().__init__()
        self.__layer1 = EltwiseMultiplicationModule(weight1)
        self.__layer2 = EltwiseMultiplicationModule(weight2)
        self.__layer3 = EltwiseMultiplicationModule(weight3)

        self.loss = L1Loss(reduction="none")

    def forward(
        self, x: Tensor, y: Tensor, use_module_trace: bool, param_prefetching: bool
    ) -> Dict[str, Tensor]:
        _assert_partition_status(
            self,
            (
                {
                    ZeroParamStatus.NOT_AVAILABLE,
                    ZeroParamStatus.INFLIGHT,
                    ZeroParamStatus.AVAILABLE,
                }
                if use_module_trace
                else {ZeroParamStatus.NOT_AVAILABLE}
            ),
        )

        pre_layer_expected_states = {
            (
                ZeroParamStatus.INFLIGHT
                if param_prefetching
                else ZeroParamStatus.NOT_AVAILABLE
            ),
            ZeroParamStatus.AVAILABLE,
        }

        post_layer_expected_states = {
            (
                ZeroParamStatus.AVAILABLE
                if param_prefetching
                else ZeroParamStatus.NOT_AVAILABLE
            ),
        }

        _assert_partition_status(self.__layer1, pre_layer_expected_states)
        hidden1 = self.__layer1(x)
        _assert_partition_status(self.__layer1, post_layer_expected_states)

        _assert_partition_status(self.__layer2, pre_layer_expected_states)
        hidden2 = self.__layer2(hidden1)
        _assert_partition_status(self.__layer2, post_layer_expected_states)

        _assert_partition_status(self.__layer3, pre_layer_expected_states)
        y_hat = self.__layer3(hidden2)
        _assert_partition_status(self.__layer3, post_layer_expected_states)

        loss = self.loss(y_hat, y)

        _assert_partition_status(
            self,
            (
                {
                    ZeroParamStatus.NOT_AVAILABLE,
                    ZeroParamStatus.INFLIGHT,
                    ZeroParamStatus.AVAILABLE,
                }
                if use_module_trace
                else {ZeroParamStatus.NOT_AVAILABLE}
            ),
        )

        return {
            "hidden1": hidden1,
            "hidden2": hidden2,
            "y_hat": y_hat,
            "loss": loss,
        }

    @staticmethod
    def to_dict(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return outputs


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
@pytest.mark.parametrize(
    "param_persistence_threshold",
    [
        0,
    ],
)
@pytest.mark.parametrize("fp16_enabled", [True, False])
@pytest.mark.parametrize(
    "contiguous_gradients",
    [
        True,
    ],
)
@pytest.mark.parametrize("offload_optimizer", [True, False])
@pytest.mark.parametrize("zero_grad", [True])
@pytest.mark.parametrize(
    "prefetching",
    [
        True,
    ],
)
@pytest.mark.parametrize("model_class", [EltwiseMultiplicationTestNetwork_Dict])
class TestZero3ParamPartitioningBase(DistributedTest):
    world_size = 2

    def test(
        self,
        param_persistence_threshold: int,
        fp16_enabled: bool,
        contiguous_gradients: bool,
        offload_optimizer: bool,
        zero_grad: bool,
        prefetching: bool,
        model_class: EltwiseMultiplicationTestNetwork_Dict,
    ) -> None:
        if offload_optimizer and not contiguous_gradients:
            return

        m = 3
        n = 5
        weights = [
            Parameter(torch.zeros((m, n), dtype=torch.float32)) for _ in range(3)
        ]
        model = model_class(*weights)
        prefetch_bucket_size = sum([p.numel() for p in model.parameters(recurse=True)])
        cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "stage3_param_persistence_threshold": param_persistence_threshold,
                "contiguous_gradients": contiguous_gradients,
                "stage3_prefetch_bucket_size": (
                    prefetch_bucket_size if prefetching else 0
                ),
            },
            "optimizer": {"type": "Adam", "params": {"lr": 1.0}},
            "fp16": {
                "enabled": fp16_enabled,
                "loss_scale": 1.0,
            },
        }

        if offload_optimizer:
            cfg["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        ds_engine = _ds_initialize_for_param_partitioning_testing(model, cfg)
        for i, weight in enumerate(weights):
            weight.ds_tensor.data = torch.full_like(
                weight.ds_tensor.data, (i + 1) * (1 + dist.get_rank())
            )

        def create_tensor(vals, dtype: torch.dtype = None) -> Tensor:
            return torch.as_tensor(
                vals,
                dtype=dtype or (torch.float16 if fp16_enabled else torch.float32),
                device=ds_engine.device,
            )

        expected_hidden1 = create_tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 2, 2],
                [2, 2, 2, 2, 2],
            ]
        )
        expected_hidden2 = create_tensor(
            [
                [2, 2, 2, 2, 2],
                [2, 2, 2, 8, 8],
                [8, 8, 8, 8, 8],
            ]
        )
        expected_yhat = create_tensor(
            [[6, 6, 6, 6, 6], [6, 6, 6, 48, 48], [48, 48, 48, 48, 48]]
        )
        expected_loss = create_tensor(
            [
                [5, 5, 5, 5, 5],
                [5, 5, 5, 47, 47],
                [47, 47, 47, 47, 47],
            ]
        )

        for train_iter in range(3):
            activations = ds_engine(
                x=torch.ones(
                    (m, n),
                    dtype=torch.float16 if fp16_enabled else torch.float32,
                    device=ds_engine.device,
                ),
                y=torch.ones(
                    (m, n),
                    dtype=torch.float16 if fp16_enabled else torch.float32,
                    device=ds_engine.device,
                ),
                use_module_trace=train_iter > 0,
                param_prefetching=prefetching and train_iter > 0,
            )
            # for ease in testing convert outputs to dict.
            activations = model_class.to_dict(activations)
            assert torch.allclose(activations["hidden1"], expected_hidden1)
            assert torch.allclose(activations["hidden2"], expected_hidden2)
            assert torch.allclose(activations["y_hat"], expected_yhat)
            assert torch.allclose(activations["loss"], expected_loss)

            ds_engine.backward(activations["loss"].sum())

            # check the gradients
            grad_partitions = ds_engine.optimizer.get_fp32_grad_partitions()
            assert set(grad_partitions.keys()) == {
                0
            }, f"should have one parameter group but got {len(grad_partitions)}"
            assert set(grad_partitions[0].keys()) == {0, 1, 2}
            dloss_wrt_layer1 = grad_partitions[0][0]
            dloss_wrt_layer2 = grad_partitions[0][1]
            dloss_wrt_layer3 = grad_partitions[0][2]

            assert dloss_wrt_layer1.dtype == torch.float
            assert dloss_wrt_layer2.dtype == torch.float
            assert dloss_wrt_layer3.dtype == torch.float

            # layer1 = [..., 1, 2, ...]
            # layer2 = [..., 2, 4, ...]
            # layer3 = [..., 3, 6, ...]
            # dloss_wrt_layer3 = hidden2
            # dloss_wrt_layer2 = layer3 * hidden1
            # dloss_wrt_layer1 = layer3 * layer2 * x

            grad_multiplier = 1 if zero_grad else (train_iter + 1)
            if dist.get_rank() == 0:
                assert torch.allclose(
                    dloss_wrt_layer3.to(get_accelerator().device_name()),
                    grad_multiplier * create_tensor([2] * 8, torch.float),
                )
                assert torch.allclose(
                    dloss_wrt_layer2.to(get_accelerator().device_name()),
                    grad_multiplier * create_tensor([3 * 1] * 8, torch.float),
                )
                assert torch.allclose(
                    dloss_wrt_layer1.to(get_accelerator().device_name()),
                    grad_multiplier * create_tensor([3 * 2 * 1] * 8, torch.float),
                )
            elif dist.get_rank() == 1:
                # parameters dont split evenly across ranks so rank 1 has a zero-padded
                # partition
                assert torch.allclose(
                    dloss_wrt_layer3.to(get_accelerator().device_name()),
                    grad_multiplier * create_tensor(([8] * 7) + [0], torch.float),
                )
                assert torch.allclose(
                    dloss_wrt_layer2.to(get_accelerator().device_name()),
                    grad_multiplier * create_tensor(([6 * 2] * 7) + [0], torch.float),
                )
                assert torch.allclose(
                    dloss_wrt_layer1.to(get_accelerator().device_name()),
                    grad_multiplier
                    * create_tensor(([6 * 4 * 1] * 7) + [0], torch.float),
                )
            else:
                raise RuntimeError("test has world size of two")

            if zero_grad:
                ds_engine.optimizer.zero_grad()

        # TODO. add testing for this - for now we just call it to make sure it
        # doesn't throw
        ds_engine.optimizer.step()
        # taking an optimizer step invalidates all parameters, make sure everything
        # has been partitioned afterwards
        _assert_partition_status(ds_engine, {ZeroParamStatus.NOT_AVAILABLE})
        assert not math.isclose(ds_engine.optimizer._global_grad_norm, 0.0)


class TestZero3InitForParentWeightInitialization(DistributedTest):
    world_size = 4

    def test(self):

        class ModelWhereParentInitializesChildWeights(Module):

            def __init__(self) -> None:
                super().__init__()

                self.linear = Linear(12, 1)

                self.apply(self.__init_weights)

            def __init_weights(self, module):
                if isinstance(module, Linear):
                    with torch.no_grad():
                        module.weight.fill_(1 + dist.get_rank())

        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {"type": "Adam", "params": {"lr": 1.0}},
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0,
            },
        }

        with deepspeed.zero.Init(
            config=ds_cfg, mem_efficient_linear=False, enabled=True
        ):
            model = ModelWhereParentInitializesChildWeights()

        assert model.linear.weight.ds_tensor.numel() == math.ceil(12 / self.world_size)
        assert torch.allclose(
            model.linear.weight.ds_tensor,
            torch.full_like(model.linear.weight.ds_tensor, 1),
        )


@pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
@pytest.mark.parametrize("force_ds_optim", [True, False])
class TestZeroOffloadOptim(DistributedTest):
    world_size = 1

    def test(self, force_ds_optim):
        config_dict = {
            "train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}},
            "zero_force_ds_cpu_optimizer": force_ds_optim,
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)

        optimizer = torch.optim.Adam(model.parameters())

        if force_ds_optim:
            with pytest.raises(ZeRORuntimeException):
                model, _, _, _ = deepspeed.initialize(
                    model=model, optimizer=optimizer, config=config_dict
                )
        else:
            model, _, _, _ = deepspeed.initialize(
                model=model, optimizer=optimizer, config=config_dict
            )
