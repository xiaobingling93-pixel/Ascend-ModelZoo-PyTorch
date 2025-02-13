/**
 * @file extension_add.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using namespace at;

// flash_attention_tik
// register forward implementation for NPU device
at::Tensor rope_mindie_sd_impl_npu(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, int64_t mode=1)
{
    at::Tensor result = at_npu::native::empty_with_format(x.sizes(),x.options(),at_npu::native::get_npu_format(x));

    at_npu::native::OpCommand cmd;

    cmd.Name("RotaryPositionEmbedding")
            .Input(x)
            .Input(cos)
            .Input(sin)
            .Output(result)
            .Attr("mode", mode)
            .Run();

    return result;
}

// register forward implementation for Meta device
at::Tensor rope_mindie_sd_impl_meta(const at::Tensor &x, const at::Tensor &cos, const at::Tensor &sin, int64_t mode)
{
    return empty_like(x);
}


// register the schemas for my_op and my_op_backward in the myops namespace
TORCH_LIBRARY(mindie, m)
{
    m.def("rope_mindie_sd(Tensor query, Tensor key, Tensor value, int mode) -> Tensor");
}

// register forward and backward implementations for the NPU device
// the device name used by the NPU device in PyTorch 2.1 and above is PrivateUse1.
// in versions below 2.1, XLA is used. If the version is below 2.1, PrivateUse1 needs to be changed to XLA.
TORCH_LIBRARY_IMPL(mindie, PrivateUse1, m)
{
    m.impl("rope_mindie_sd", &rope_mindie_sd_impl_npu);
}

// bind the NPU's autograd implementation to the operation
// if the version is below PyTorch 2.1, AutogradPrivateUse1 needs to be changed to AutogradXLA.

// register forward and backward implementations for the Meta device
TORCH_LIBRARY_IMPL(mindie, Meta, m)
{
    m.impl("rope_mindie_sd", &rope_mindie_sd_impl_meta);
}