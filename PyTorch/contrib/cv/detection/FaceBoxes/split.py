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

import os
import shutil
path = os.getcwd()
pre_dir = os.path.join(path, 'convert')
cur_path = os.path.join(path, 'pred_sample')
target_path = "compare_data"

if not os.path.exists(target_path):
    os.mkdir(target_path)

for dir_name in os.listdir(cur_path):
    tmp_path = os.path.join(cur_path, dir_name.strip('\n'))
    target_dir = os.path.join(target_path, dir_name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for data in os.listdir(tmp_path):
        pre_file = os.path.join(pre_dir, data)
        cur_file = os.path.join(tmp_path, data)
        if os.path.exists(pre_file):
            target_file = os.path.join(target_dir, data)
            shutil.copy(pre_file, target_file)
