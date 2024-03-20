# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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


from argparse import ArgumentParser

from auto_optimizer import OnnxGraph


def main():
    # 目前AiCPU算子ReduceMax不支持axes传空取默认值，必须显式输入axes。
    # 否则会导致atc编译失败。

    parser = ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    args = parser.parse_args()

    graph = OnnxGraph.parse(args.onnx)
    reduceMax = graph.get_nodes('ReduceMax')[0]
    reduceMax.attrs['axes'] = [0]

    graph.update_map()
    graph.infershape()

    g_sim = graph.simplify()
    save_path = args.onnx.replace(".onnx", "_modified.onnx")
    g_sim.save(save_path)
    print("Modified model saved to ", save_path)


if __name__ == "__main__":
    main()