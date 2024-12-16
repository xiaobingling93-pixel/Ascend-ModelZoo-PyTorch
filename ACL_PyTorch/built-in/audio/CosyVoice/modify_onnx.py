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

import os
import sys

from auto_optimizer import OnnxGraph


def modify_camplus(camplus):
    p2 = {'ceil_mode': 1, 'kernel_shape': [2], 'pads': [0, 0], 'strides': [2]}
    pool_list = camplus.get_nodes('AveragePool')
    for pool in pool_list:
        camplus.add_node(f"{pool.name}_2", 'AveragePool', inputs=[f"{pool.name}_out1"], outputs=[pool.outputs[0]], attrs=p2)
        pool.outputs[0] = f"{pool.name}_out1"
        pool.attrs['kernel_shape'] = [50]
        pool.attrs['strides'] = [50]
    camplus.save('./campplus_md.onnx')


def modify_speech_token(speech_token):
    speech_token['/ReduceMax'].attrs['keepdims'] = 1
    speech_token.remove('/Unsqueeze_2')
    speech_token.save('./speech_token_md.onnx')


if __name__ == '__main__':
    input_path = sys.argv[1]
    camplus = OnnxGraph.parse(os.path.join(input_path, "campplus.onnx"))
    # 当前om中的Averagepool算子不支持超过50的kernel shape，改图分解算子
    modify_camplus(camplus)
    speech_token = OnnxGraph.parse(os.path.join(input_path, "speech_tokenizer_v1.onnx"))
    # om图模式不支持无shape输出，reducemax需要修改keepdim
    modify_speech_token(speech_token)
