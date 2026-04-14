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


import os
import subprocess
import lerobot


def main():
    lerobot_path = lerobot.__path__

    subprocess.check_call(
        ["git", "apply", "--whitespace=fix", os.path.abspath("lerobot_diff.patch")],
        cwd=lerobot_path[0],
    )
    print("lerobot patch applied successfully.")


if __name__ == '__main__':
    main()


