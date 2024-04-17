#!/bin/bash

# 设置modelzoo项目下cogvlm_utils路径，例如"xxx/cogvlm_utils"，用户需要自行设定
MODEL_ZOO_SOURCE_DIR="model_zoo路径"
# 设置CogVLM项目路径，例如"xxx/CogVLM-main"，用户需要自行设定
COGVLM_SOURCE_DIR="Cogvlm路径"
# 设置三方件sat路径，为conda环境下三方件安装路径，例如"xxx/python3.8/site-packages/sat"，用户需要自行设定
SAT_SOURCE_DIR="sat路径"

# 定义一个函数，用于给文件添加modelzoobackup后缀
rename_file() {
    local ORIGINAL_FILE_PATH=$1
    local BACKUP_FILE_PATH

    # 提取原始文件的基本名称和目录路径
    local FILE_NAME=$(basename "$ORIGINAL_FILE_PATH")
    local DIRECTORY_PATH=$(dirname "$ORIGINAL_FILE_PATH")

    # 构建带有backup后缀的新文件名
    local BACKUP_FILE_NAME="${FILE_NAME}.modelzoobackup"

    # 构建新文件的完整路径
    BACKUP_FILE_PATH="$DIRECTORY_PATH/$BACKUP_FILE_NAME"

    # 检查原始文件是否存在
    if [ ! -f "$ORIGINAL_FILE_PATH" ]; then
        echo "Error: Original file '$ORIGINAL_FILE_PATH' does not exist."
        return 1
    fi

    # 检查备份文件是否已经存在
    if [ -e "$BACKUP_FILE_PATH" ]; then
        echo "Warning: Backup file '$BACKUP_FILE_PATH' already exists."
        # 如果需要覆盖备份文件，可以取消注释下面的行
        # rm "$BACKUP_FILE_PATH"
    fi

    # 使用mv命令创建备份文件
    mv "$ORIGINAL_FILE_PATH" "$BACKUP_FILE_PATH"
    echo "Backup created: $BACKUP_FILE_PATH (from $ORIGINAL_FILE_PATH)"
}

# 定义拷贝文件的函数
copy_file() {
    local SOURCE_FILE=$1
    local DESTINATION_FILE=$2

    # 检查源文件是否存在
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "错误：源文件 '$SOURCE_FILE' 不存在。"
        return 1
    fi

    # 使用cp命令拷贝文件
    cp "$SOURCE_FILE" "$DESTINATION_FILE"

    # 检查拷贝是否成功
    if [ $? -eq 0 ]; then
        echo "文件拷贝成功，从 '$SOURCE_FILE' 到 '$DESTINATION_FILE'。"
    else
        echo "错误：文件拷贝失败。"
    fi
}

# 检查源目录是否存在
if [ ! -d "$MODEL_ZOO_SOURCE_DIR" ]; then
    echo "Error: Source directory $MODEL_ZOO_SOURCE_DIR does not exist."
    exit 1
fi

if [ ! -d "$COGVLM_SOURCE_DIR" ]; then
    echo "Error: Source directory $COGVLM_SOURCE_DIR does not exist."
    exit 1
fi

if [ ! -d "$SAT_SOURCE_DIR" ]; then
    echo "Error: Source directory $SAT_SOURCE_DIR does not exist."
    exit 1
fi

# model_zoo path
eva_clip_model_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/eva_clip_model.py"
mixin_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/mixin.py"
finetune_cogvlm_demo_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/finetune_cogvlm_demo.py"
triton_rotary_embeddings_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/triton_rotary_embeddings.py"
transformer_defaults_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/transformer_defaults.py"
finetune_cogvlm_base_224_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/finetune_cogvlm_base_224.sh"
layernorm_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/layernorm.py"
dataset_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/dataset.py"
evaluate_cogvlm_demo_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/code/evaluate_cogvlm_demo.py"
eval_cogvlm_base_224_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/eval_cogvlm_base_224.sh"
env_npu_model_zoo_path="$MODEL_ZOO_SOURCE_DIR/env_npu.sh"

# cogvlm path
eva_clip_model_cogvlm_path="$COGVLM_SOURCE_DIR/utils/models/eva_clip_model.py"
mixin_cogvlm_path="$COGVLM_SOURCE_DIR/utils/models/mixin.py"
dataset_cogvlm_path="$COGVLM_SOURCE_DIR/utils/utils/dataset.py"
finetune_cogvlm_demo_cogvlm_path="$COGVLM_SOURCE_DIR/finetune_demo/finetune_cogvlm_demo.py"
evaluate_cogvlm_demo_cogvlm_path="$COGVLM_SOURCE_DIR/finetune_demo/evaluate_cogvlm_demo.py"
finetune_demo_cogvlm_path="$COGVLM_SOURCE_DIR/finetune_demo/"

# sat path
triton_rotary_embeddings_sat_path="$SAT_SOURCE_DIR/model/position_embedding/triton_rotary_embeddings.py"
transformer_defaults_sat_path="$SAT_SOURCE_DIR/transformer_defaults.py"
layernorm_sat_path="$SAT_SOURCE_DIR/ops/layernorm.py"

# cogvlm文件改名
rename_file $eva_clip_model_cogvlm_path
rename_file $mixin_cogvlm_path
rename_file $finetune_cogvlm_demo_cogvlm_path
rename_file $evaluate_cogvlm_demo_cogvlm_path
rename_file $dataset_cogvlm_path

# sat文件改名
rename_file $triton_rotary_embeddings_sat_path
rename_file $transformer_defaults_sat_path
rename_file $layernorm_sat_path

# cogvlm文件替换
copy_file $eva_clip_model_model_zoo_path $(dirname "$eva_clip_model_cogvlm_path")
copy_file $mixin_model_zoo_path $(dirname "$mixin_cogvlm_path")
copy_file $finetune_cogvlm_demo_model_zoo_path $(dirname "$finetune_cogvlm_demo_cogvlm_path")
copy_file $evaluate_cogvlm_demo_model_zoo_path $(dirname "$evaluate_cogvlm_demo_cogvlm_path")
copy_file $dataset_model_zoo_path $(dirname "$dataset_cogvlm_path")
copy_file $finetune_cogvlm_base_224_model_zoo_path $finetune_demo_cogvlm_path
copy_file $eval_cogvlm_base_224_model_zoo_path $finetune_demo_cogvlm_path
copy_file $env_npu_model_zoo_path $finetune_demo_cogvlm_path

# sat文件替换
copy_file $triton_rotary_embeddings_model_zoo_path $(dirname "$triton_rotary_embeddings_sat_path")
copy_file $transformer_defaults_model_zoo_path $(dirname "$transformer_defaults_sat_path")
copy_file $layernorm_model_zoo_path $(dirname "$layernorm_sat_path")

echo "All files copied successfully."