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


import sys

import torch
import mindietorch

ENCODER_X_SHAPE = (1, 100, 80)
ENCODER_X_LENS_SHAPE = (1, )

inputs = [mindietorch.Input(ENCODER_X_SHAPE, dtype=torch.float32), mindietorch.Input(ENCODER_X_LENS_SHAPE, dtype=torch.int64)]

encoder_ts_model = torch.jit.load('./exp/exported_encoder-epoch-99-avg-1.ts')
encoder_ts_model.eval()

mindietorch.set_device(0)

try:
    compiled_encoder_model = mindietorch.compile(
        encoder_ts_model,
        inputs=inputs,
        precision_policy=mindietorch.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
    )
    compiled_encoder_model.save("./compiled_encoder.ts")
except Exception as e:
    print("an error has occured.")
    print(e)
    sys.exit(1)
