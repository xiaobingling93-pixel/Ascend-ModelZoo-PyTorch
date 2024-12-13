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
sys.path.append("./FunASR")

import torch
import mindietorch

from mindie_paraformer import precision_eval


def cif(hidden, alphas, integrate, frame, threshold):
    batch_size, len_time, hidden_size = hidden.size()

    # intermediate vars along time
    list_fires = []
    list_frames = []

    constant = torch.ones([batch_size], device=hidden.device)
    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = constant - integrate

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place, integrate - constant, integrate
        )
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(
            fire_place[:, None].repeat(1, hidden_size), remainds[:, None] * hidden[:, t, :], frame
        )

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)

    return fires, frames, integrate, frame


def cif_wo_hidden(alphas, integrate, threshold):
    batch_size, len_time = alphas.size()

    list_fires = []

    constant = torch.ones([batch_size], device=alphas.device) * threshold

    for t in range(len_time):
        alpha = alphas[:, t]

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - constant,
            integrate,
        )

    fire_list = []
    for i in range(0, len(list_fires), 500):
        batch = list_fires[i:i + 500]
        fire = torch.stack(batch, 1)
        fire_list.append(fire)
    
    fires = torch.cat(fire_list, 1)
    return fires, integrate


class MindieCif(torch.nn.Module):
    def __init__(self, threshold, seq_len):
        super().__init__()
        self.threshold = threshold
        self.seq_len = seq_len
    
    def forward(self, hidden, alphas, integrate, frame):
        fires, frames, integrate_new, frame_new = cif(hidden, alphas, integrate, frame, self.threshold)

        frame = torch.index_select(frames[0, :, :], 0, torch.nonzero(fires[0, :] >= self.threshold).squeeze(1))

        return frame, integrate_new, frame_new

    def export_ts(self, path="./compiled_cif.pt", soc_version="Ascendxxx"):
        print("Begin tracing cif function.")

        input_shape1 = (1, self.seq_len, 512)
        input_shape2 = (1, self.seq_len)
        input_shape3 = (1, )
        input_shape4 = (1, 512)

        hidden = torch.randn(input_shape1, dtype=torch.float32)
        alphas = torch.randn(input_shape2, dtype=torch.float32)
        integrate = torch.randn(input_shape3, dtype=torch.float32)
        frame = torch.randn(input_shape4, dtype=torch.float32)
        compile_inputs = [mindietorch.Input(shape=input_shape1, dtype=torch.float32),
                          mindietorch.Input(shape=input_shape2, dtype=torch.float32),
                          mindietorch.Input(shape=input_shape3, dtype=torch.float32),
                          mindietorch.Input(shape=input_shape4, dtype=torch.float32)]
        
        export_model = torch.jit.trace(self, example_inputs=(hidden, alphas, integrate, frame))
        print("Finish tracing cif function.")

        compiled_model = mindietorch.compile(
            export_model,
            inputs=compile_inputs,
            precision_policy=mindietorch.PrecisionPolicy.PREF_FP16,
            default_buffer_size_vec=[1, 10, 10],
            soc_version=soc_version,
            ir="ts"
        )
        compiled_model.save(path)
        print("Finish compiling cif function, compiled model is saved in {}.".format(path))

        print("Start checking the percision of cif function.")
        sample_hidden = torch.randn(input_shape1, dtype=torch.float32)
        sample_alphas = torch.randn(input_shape2, dtype=torch.float32)
        sample_integrate = torch.randn(input_shape3, dtype=torch.float32)
        sample_frame = torch.randn(input_shape4, dtype=torch.float32)
        mrt_res = compiled_model(sample_hidden.to("npu"), sample_alphas.to("npu"),
                                 sample_integrate.to("npu"), sample_frame.to("npu"))
        ref_res = self.forward(sample_hidden, sample_alphas, sample_integrate, sample_frame)
        precision_eval(mrt_res, ref_res)


class MindieCifTimestamp(torch.nn.Module):
    def __init__(self, threshold, seq_len):
        super().__init__()
        self.threshold = threshold
        self.seq_len = seq_len
    
    def forward(self, us_alphas, integrate):
        us_peaks, integrate_new = cif_wo_hidden(us_alphas, integrate, self.threshold)

        return us_peaks, integrate_new

    def export_ts(self, path="./compiled_cif_timestamp.ts", soc_version="Ascendxxx"):
        print("Begin tracing cif_timestamp function.")

        input_shape1 = (1, self.seq_len)
        input_shape2 = (1, )

        us_alphas = torch.randn(input_shape1, dtype=torch.float32)
        integrate = torch.randn(input_shape2, dtype=torch.float32)
        compile_inputs = [mindietorch.Input(shape=input_shape1, dtype=torch.float32),
                          mindietorch.Input(shape=input_shape2, dtype=torch.float32)]
        
        export_model = torch.jit.trace(self, example_inputs=(us_alphas, integrate))
        print("Finish tracing cif_timestamp function.")

        compiled_model = mindietorch.compile(
            export_model,
            inputs=compile_inputs,
            precision_policy=mindietorch.PrecisionPolicy.PREF_FP16,
            default_buffer_size_vec=[1, 10],
            soc_version=soc_version,
            ir="ts"
        )
        compiled_model.save(path)
        print("Finish compiling cif_timestamp function, compiled model is saved in {}.".format(path))

        print("Start checking the percision of cif_timestamp function.")
        sample_input1 = torch.randn(input_shape1, dtype=torch.float32)
        sample_input2 = torch.randn(input_shape2, dtype=torch.float32)
        mrt_res = compiled_model(sample_input1.to("npu"), sample_input2.to("npu"))
        ref_res = self.forward(sample_input1, sample_input2)
        precision_eval(mrt_res, ref_res)