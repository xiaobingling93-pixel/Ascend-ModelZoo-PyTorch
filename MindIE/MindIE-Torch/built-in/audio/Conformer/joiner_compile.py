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

JOINER_X_SHAPE = (1, 512)

inputs = [mindietorch.Input(JOINER_X_SHAPE, dtype=torch.float32), mindietorch.Input(JOINER_X_SHAPE, dtype=torch.float32)]

joiner_ts_model = torch.jit.load('./exp/exported_joiner-epoch-99-avg-1.ts')
joiner_ts_model.eval()
mindietorch.set_device(0)

try:
    compiled_joiner = mindietorch.compile(
        joiner_ts_model,
        inputs=inputs,
        precision_policy=mindietorch.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        soc_version="Ascend310P3",
    )
    compiled_joiner.save("./compiled_joiner.ts")
except Exception as e:
    print("During the compilation of joiner model, an error has occured.")
    import sys
    sys.exit(1)
