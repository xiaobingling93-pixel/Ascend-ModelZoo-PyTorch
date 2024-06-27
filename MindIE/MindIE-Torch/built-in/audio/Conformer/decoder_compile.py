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

DECODER_Y_SHAPE = (1, 2)

inputs = [mindietorch.Input(DECODER_Y_SHAPE, dtype=torch.int64)]

decoder_ts_model = torch.jit.load('./exp/exported_decoder-epoch-99-avg-1.ts')
decoder_ts_model.eval()
mindietorch.set_device(0)
try:
    compiled_decoder = mindietorch.compile(
        decoder_ts_model,
        inputs=inputs,
        precision_policy=_enums.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        soc_version="Ascend310P3",
    )
    compiled_decoder.save("compiled_decoder.ts")
except Exception as e:
    print(f"During the compilation of decoder model, an error has occured: {e}")

    import sys
    sys.exit(1)
