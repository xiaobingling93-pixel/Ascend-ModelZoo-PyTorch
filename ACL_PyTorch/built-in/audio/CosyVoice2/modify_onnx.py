# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
from auto_optimizer import OnnxGraph


def modify_speech_token(speech_token):
    ReduceMax_list = speech_token.get_nodes("ReduceMax")
    for node in ReduceMax_list:
        if node.attrs['keepdims'] == 0:
            out_nodes = speech_token.get_next_nodes(node.outputs[0])
            for out_node in out_nodes:
                if out_node.op_type == "Unsqueeze":
                    node.attrs['keepdims'] = 1
                    speech_token.remove(out_node.name)
    return speech_token

if __name__ == '__main__':
    input_path = sys.argv[1]
    speech_token = OnnxGraph.parse(os.path.join(input_path, "speech_tokenizer_v2.onnx"))
    # om图模式不支持无shape输出，reducemax需要修改keepdim
    modify_onnx = modify_speech_token(speech_token)
    modify_onnx.save(os.path.join(input_path, "speech_token_md.onnx"))