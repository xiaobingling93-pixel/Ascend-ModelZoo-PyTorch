#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

BUILD_DIR=$(dirname "$(realpath "$0")")
CODE_ROOT=$(dirname "${BUILD_DIR}")
WEIGHT_DOWNLOAD_SCRIPT_PATH="$BUILD_DIR/download_weights.py"
WEIGHT_DIR=$(realpath "${CODE_ROOT}/..")
atb_models_tar_dir=$(realpath "${CODE_ROOT}/../..")

function fn_prepare_weights() {
    python3 "$WEIGHT_DOWNLOAD_SCRIPT_PATH" --target_dir "$WEIGHT_DIR"
}

function fn_prepare_atb_models() {
    local atb_models_tar_pattern="Ascend-mindie-atb-models_*.tar.gz"
    local atb_models_tar_path=""
    
    if find "${atb_models_tar_dir}" -maxdepth 1 -name "${atb_models_tar_pattern}"; then
        atb_models_tar_path=$(find "${atb_models_tar_dir}" -maxdepth 1 -name "${atb_models_tar_pattern}")
        echo "Found Ascend MindIE ATB Models tar file: ${atb_models_tar_path}"
        unpacked_dir=${atb_models_tar_dir}/llm_model
        if [[ -d "${unpacked_dir}" ]]; then
            rm -rf $unpacked_dir
        fi
        mkdir -p $unpacked_dir
        tar -xzvf "${atb_models_tar_path}" -C "${unpacked_dir}"

        if [[ ! -d "${unpacked_dir}" ]]; then
            echo "Ascend MindIE ATB Models Unpacked directory not found."
            exit 1
        fi

        local atb_framework_dir="${unpacked_dir}/atb_framework/"

        if [[ ! -d "${atb_framework_dir}" ]]; then
            echo "Ascend MindIE ATB Models atb_framework dir not found."
            exit 1
        fi

        cp -r "$CODE_ROOT/atb_framework/"* "$atb_framework_dir/models"

        local atb_llm_dir="${unpacked_dir}/atb_llm/"

        if [[ ! -d "${atb_llm_dir}" ]]; then
            echo "Ascend MindIE ATB Models atb_llm dir not found."
            exit 1
        fi

        cp -r "$atb_llm_dir/models/base" "$CODE_ROOT/atb_llm/models"

        echo "atb_models files copied successfully."
        echo "Ascend MindIE ATB Models is successfully prepared."
    else
        echo "Missing Ascend MindIE ATB Models tar file."
        exit 1
    fi
}

fn_prepare_third_party()
{
    local nlohmann_path="${atb_models_tar_dir}/include.zip"

    if [[ ! -e "${nlohmann_path}" ]]; then
        echo "Local nlohmannJson zip file not found. Will try to download later."
        return
    fi

    mkdir -p $unpacked_dir/build/nlohmann

    cp -f "$nlohmann_path" "$unpacked_dir/build/nlohmann"

    echo "nlohmannJson zip files copied successfully."
    echo "Ascend MindIE ATB Models third-party is successfully prepared."
}

function fn_build()
{
    bash "${unpacked_dir}/scripts/modelers_build.sh"
}

function fn_main()
{
    fn_prepare_weights
    fn_prepare_atb_models
    fn_prepare_third_party
    fn_build
}

fn_main "$@"