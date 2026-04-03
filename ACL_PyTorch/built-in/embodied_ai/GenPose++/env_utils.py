# Copyright 2026 Huawei Technologies Co., Ltd
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

import subprocess


def execute_command(cmd_list):
    with subprocess.Popen(cmd_list,
                          shell=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        out, err = p.communicate(timeout=1000)
    res = out.decode()
    return res


def is_rc_device():
    lspci_output = execute_command(["lspci"])
    pci_info = [line for line in lspci_output.split("\n") if "accelerators" in line.strip()]
    return not pci_info