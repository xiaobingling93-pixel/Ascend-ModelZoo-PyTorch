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
import onnx
import onnx.helper as helper 
from onnx import TensorProtos

model = onnx.load("qwen2.onnx")
new_nodes = []

for node in model.graph.node:
    new_nodes = node
    if node.op_tyoe == "Trilu":
        new_node = helper.make_node(
            "Trilu",
            inputs=[node.input[0]],
            outputs=node.output,
            upper=1
        )
    new_nodes.append(new_node)
    
new_graph = helper.make_graph(
    new_nodes,
    "new_graph",
    inputs=model.graph.input,
    outputs=model.graph.output,
    value_info=model.graph.value_info,
    initializer=model.graph.initializer
)

new_model = helper.make_model(new_graph, producer_name=model.producer_name, opset_imports=model.opset_import, ir_version=model.ir_version)
onnx.save(new_model, "qwen2.onnx", save_as_external_data=True)