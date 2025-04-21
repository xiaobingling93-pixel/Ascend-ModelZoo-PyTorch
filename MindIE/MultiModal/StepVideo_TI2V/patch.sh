cp -r xDiT/xfuser .

patch_path="stepvideo/utils/parallel_utils/"
declare -A files=(
    ["envs.py"]="xfuser/envs.py"
    ["xfuser_init.py"]="xfuser/__init__.py"
    ["models_init.py"]="xfuser/model_executor/models/__init__.py"
    ["core_init.py"]="xfuser/core/__init__.py"
    ["ring_flash_attn.py"]="xfuser/core/long_ctx_attention/ring/ring_flash_attn.py"
    ["config.py"]="xfuser/config/config.py"
    ["tp_applicator.py"]="xfuser/model_executor/models/customized/step_video_t2v/tp_applicator.py"
    ["linear.py"]="xfuser/model_executor/models/customized/step_video_t2v/linear.py"
    ["attn_layer.py"]="xfuser/core/long_ctx_attention/hybrid/attn_layer.py"
)

for src in "${!files[@]}"; do
    cp -r "${patch_path}${src}" "${files[$src]}"
done
