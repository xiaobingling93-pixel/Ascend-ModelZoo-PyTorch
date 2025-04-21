// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <iostream>

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "atb/infer_op_params.h"
#endif

using namespace std;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
namespace {


void MatmulAllreduce_Operation(const at::Tensor &x, const at::Tensor &weight, at::Tensor &output, const int rank, const int ranksize)
{
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "ATB MatmulAll_reduce is not implemented");
#else
        atb::infer::LinearParallelParam param;
        param.transWeight = true;
        param.rank = rank;
        param.rankSize = ranksize;
        param.hasResidual = false;
        param.backend = "lcoc";
        param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;

        ParamSetter paramsetter;
        paramsetter.Input(x);
        paramsetter.Input(weight);
        paramsetter.Output(output);
        atb::Operation* op = nullptr;
        atb::Status st = atb::CreateOperation(param, &op);

        TORCH_CHECK(st == atb::NO_ERROR && op, "MatmulAllReduceOperation get op failed!");
        RunAtbCmd(op, paramsetter, "MatmulAllReduce_Operation");

        return;
#endif
}
} //namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_allreduce", &MatmulAllreduce_Operation, "Matmul_AllReduce on ascend device",
            pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("output"), pybind11::arg("rank"), pybind11::arg("ranksize"));
}
