# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.append("./FunASR")

import torch
import mindietorch

from mindie_paraformer import precision_eval


class MindieVAD(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.encoder.eval()
        for para in model.encoder.parameters():
            para.requires_grad = False
        self.model = model
    
    def forward(self, feat):
        result = self.model.encoder(feat, {})
        return result

    @staticmethod
    def export(vad, path="./compiled_vad.pt", soc_version="Ascendxxx"):
        print("Begin tracing vad model.")
        input_shape = (1, 5996, 400)
        min_shape = (1, -1, 400)
        max_shape = (1, -1, 400)
        input_feat = torch.randn(input_shape, dtype=torch.float32)
        compile_inputs = [mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=torch.float32)]
        
        export_model = torch.jit.trace(vad, input_feat)
        print("Finish tracing vad model.")

        compiled_model = mindietorch.compile(
            export_model,
            inputs=compile_inputs,
            precision_policy=mindietorch.PrecisionPolicy.PREF_FP32,
            default_buffer_size_vec=[50, ],
            soc_version=soc_version,
            ir="ts"
        )
        compiled_model.save(path)
        print("Finish compiling vad model, compiled model is saved in {}.".format(path))

        print("Start checking the percision of vad model.")
        sample_feat = torch.randn(input_shape, dtype=torch.float32)
        mrt_res = compiled_model(sample_feat.to("npu"))
        ref_res = vad(sample_feat)
        precision_eval(mrt_res, ref_res)
