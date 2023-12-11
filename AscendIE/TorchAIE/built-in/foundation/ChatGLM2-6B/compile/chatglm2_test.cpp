/* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

*/

#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>

#include "torch_aie.h"

static const int deviceId = 0;
namespace {
const std::string MODULE_DIR = "../chatglm2_6b_batch_1_traced.pt";
const std::string TORCHAIE_MODULE_DIR = "../chatglm2_6b_batch_1_compiled.pt";
const int MAX_SEQLEN = 32768;

auto getCompileSpec() -> torch_aie::torchscript::CompileSpec
{
    int batch = 1;
    std::vector<int64_t> Shape1Min = {1, 1}; // input_id
    std::vector<int64_t> Shape1Max = {batch, MAX_SEQLEN};
    std::vector<int64_t> Shape2Min = {1, 1}; // position_id
    std::vector<int64_t> Shape2Max = {batch, MAX_SEQLEN};
    std::vector<int64_t> Shape1Min = {1, 1}; // attention_mask
    std::vector<int64_t> Shape1Max = {batch, MAX_SEQLEN};
    std::vector<int64_t> Shape1Min = {1, 2, 0, 1, 2, 128}; // past_key_values
    std::vector<int64_t> Shape1Max = {28, 2, MAX_SEQLEN, batch, 2, 128};

    //dynamic shape
    std::vector<torch_aie::Input> inputs;
    inputs.emplace_back(torch_aie::Input(Shape1Min, Shape1Max, torch_aie::DataType::INT64,
        torch_aie::TensorFormat::NCHW));
    inputs.emplace_back(torch_aie::Input(Shape2Min, Shape2Max, torch_aie::DataType::INT64,
        torch_aie::TensorFormat::NCHW));
    inputs.emplace_back(torch_aie::Input(Shape3Min, Shape3Max, torch_aie::DataType::INT64,
        torch_aie::TensorFormat::NCHW));
    inputs.emplace_back(torch_aie::Input(Shape4Min, Shape4Max, torch_aie::DataType::FLOAT,
        torch_aie::TensorFormat::NCHW));
    
    torch_aie::torchscript::CompileSpec compileSpec(inputs);
    std::string soc_version = "Ascend910B4";
    compileSpec.soc_version = soc_version;
    compileSpec.allow_tensor_replace_int = true;
    compileSpec

}

}