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

import subprocess
import logging
import diffusers


def main():
    diffusers_path = diffusers.__path__
    diffusers_version = diffusers.__version__

    assert diffusers_version == '0.29.0', "expectation diffusers==0.29.0"
    result = subprocess.run(["patch", "-p0", f"{diffusers_path[0]}/models/attention_processor.py",
                             "attention_processor.patch"], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("Patch failed, error message: s%", result.stderr)


if __name__ == '__main__':
    main()
