#!/bin/bash
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
set -e

if command -v python3 &> /dev/null; then
    python_command=python3
else
    python_command=python
fi

pip install coverage
pip install colossalai==0.4.4 --no-deps
pip install pytest
pip install pytest-cov

current_directory=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=${current_directory}/../:$PYTHONPATH

pytest -k "test_ and not _test" --cov=../opensoraplan --cov-branch --cov-report xml --cov-report html \
--junit-xml=${current_directory}/final.xml \
--continue-on-collection-errors