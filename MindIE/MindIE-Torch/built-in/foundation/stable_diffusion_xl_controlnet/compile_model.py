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

import torch
import mindietorch
from mindietorch import _enums

def compile_clip(model, inputs, clip_compiled_path, soc_version):
    compiled_clip_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=False,
                            truncate_long_and_double=False,
                            min_block_size = 1,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            soc_version=soc_version,
                            optimization_level=0))
    torch.jit.save(compiled_clip_model, clip_compiled_path)

def compile_vae(model, inputs, vae_compiled_path, soc_version):
    compiled_vae_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=True,
                            truncate_long_and_double=True,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0
                            ))
    torch.jit.save(compiled_vae_model, vae_compiled_path)


def compile_unet(model, inputs, unet_compiled_path, soc_version):
    compiled_unet_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=True,
                            truncate_long_and_double=True,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0
                            ))
    torch.jit.save(compiled_unet_model, unet_compiled_path)

def compile_control(model, inputs, control_compiled_path, soc_version):
    compiled_control_model = (
        mindietorch.compile(model,
                            inputs=inputs,
                            allow_tensor_replace_int=True,
                            require_full_compilation=False,
                            truncate_long_and_double=False,
                            soc_version=soc_version,
                            precision_policy=_enums.PrecisionPolicy.FP16,
                            optimization_level=0
                            ))
    torch.jit.save(compiled_control_model, control_compiled_path)