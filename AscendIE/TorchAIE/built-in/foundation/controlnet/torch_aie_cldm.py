# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import torch
import torch_aie
from torch_aie import _enums
from cldm.cldm import ControlLDM


class TorchAIEControlLDM(ControlLDM):
    device_0 = None

    def compile_pt_model(self, control_path, sd_path):
        if not os.path.exists(control_path):
            print(f"the control model file does not exist in path: {control_path}!")
        else:
            control_pt = torch.jit.load(control_path).eval()

        if not os.path.exists(sd_path):
            print(f"the sd model file does not exist in path: {sd_path}!")
        else:
            sd_pt = torch.jit.load(sd_path).eval()
        control_input_info = [
            torch_aie.Input((1, 4, 64, 72), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 3, 512, 576), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1,), dtype=torch_aie.dtype.INT32),
            torch_aie.Input((1, 77, 768), dtype=torch_aie.dtype.FLOAT),
        ]
        self.compiled_control_model = torch_aie.compile(
            control_pt,
            inputs=control_input_info,
            allow_tensor_replace_int=True,
            truncate_long_and_double=True,
            soc_version="Ascend910B3",
            precision_policy=_enums.PrecisionPolicy.FP16,
            optimization_level=0,
        )
        torch.jit.save(self.compiled_control_model, control_path[:-3] + "_compiled.pt")

        sd_input_info = [
            torch_aie.Input((1, 4, 64, 72), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1,), dtype=torch_aie.dtype.INT32),
            torch_aie.Input((1, 77, 768), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 320, 64, 72), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 320, 64, 72), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 320, 64, 72), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 320, 32, 36), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 640, 32, 36), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 640, 32, 36), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 640, 16, 18), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 1280, 16, 18), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 1280, 16, 18), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 1280, 8, 9), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 1280, 8, 9), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 1280, 8, 9), dtype=torch_aie.dtype.FLOAT),
            torch_aie.Input((1, 1280, 8, 9), dtype=torch_aie.dtype.FLOAT),
        ]
        self.compiled_sd_model = torch_aie.compile(
            sd_pt,
            inputs=sd_input_info,
            allow_tensor_replace_int=True,
            truncate_long_and_double=True,
            soc_version="Ascend910B3",
            precision_policy=_enums.PrecisionPolicy.FP16,
            optimization_level=0,
        )
        torch.jit.save(self.compiled_sd_model, sd_path[:-3] + "_compiled.pt")

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        if cond["c_concat"] is None:
            eps = self.compiled_sd_model(x_noisy, t, cond_txt)
        else:
            control = self.compiled_control_model(
                x_noisy.to(f"npu:{self.device_0}"),
                torch.cat(cond["c_concat"], 1).to(f"npu:{self.device_0}"),
                t.int().to(f"npu:{self.device_0}"),
                cond_txt.to(f"npu:{self.device_0}"),
            )
            control = [
                c.to("cpu") * scale for c, scale in zip(control, self.control_scales)
            ]
            eps = self.compiled_sd_model(
                x_noisy.to(f"npu:{self.device_0}"),
                t.int().to(f"npu:{self.device_0}"),
                cond_txt.to(f"npu:{self.device_0}"),
                control[0].to(f"npu:{self.device_0}"),
                control[1].to(f"npu:{self.device_0}"),
                control[2].to(f"npu:{self.device_0}"),
                control[3].to(f"npu:{self.device_0}"),
                control[4].to(f"npu:{self.device_0}"),
                control[5].to(f"npu:{self.device_0}"),
                control[6].to(f"npu:{self.device_0}"),
                control[7].to(f"npu:{self.device_0}"),
                control[8].to(f"npu:{self.device_0}"),
                control[9].to(f"npu:{self.device_0}"),
                control[10].to(f"npu:{self.device_0}"),
                control[11].to(f"npu:{self.device_0}"),
                control[12].to(f"npu:{self.device_0}"),
            ).to("cpu")

        return eps
