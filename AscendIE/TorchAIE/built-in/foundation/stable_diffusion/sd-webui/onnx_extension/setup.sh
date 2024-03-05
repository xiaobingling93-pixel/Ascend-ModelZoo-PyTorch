#! /bin/bash

soc_version=$(echo $(npu-smi info) | cut -d "|" -f12 | cut -d " " -f3)

current_path=$(cd $(dirname $0);pwd)

python ${current_path}/export2onnx.py
python ${current_path}/modify_onnx.py

model_clip="models/SD2.1/models_bs1/clip"
model_unet="models/SD2.1/models_bs1/unet"
model_vae="models/SD2.1/models_bs1/vae"

# clip
if [ ! -e "${current_path}/${model_clip}/clip.om" ]; then
    atc --framework=5 \
        --model=${current_path}/${model_clip}/clip.onnx \
        --output=${current_path}/${model_clip}/clip \
        --input_format=ND \
        --log=error \
        --soc_version=Ascend${soc_version}
fi

# unet
if [ ! -e "${current_path}/${model_unet}/unet.om" ]; then
    atc --framework=5 \
        --model=${current_path}/${model_unet}/unet_md.onnx \
        --output=${current_path}/${model_unet}/unet \
        --input_format=NCHW \
        --log=error \
        --optypelist_for_implmode="Gelu,Sigmoid" \
        --op_select_implmode=high_performance \
        --soc_version=Ascend${soc_version}
fi

# vae
if [ ! -e "${current_path}/${model_vae}/vae.om" ]; then
    atc --framework=5 \
        --model=${current_path}/${model_vae}/vae.onnx \
        --output=${current_path}/${model_vae}/vae \
        --input_format=NCHW \
        --log=error \
        --soc_version=Ascend${soc_version}
fi

model_parallel_clip="models/SD2.1/models_bs1_parallel/clip"
model_parallel_unet="models/SD2.1/models_bs1_parallel/unet"
model_parallel_vae="models/SD2.1/models_bs1_parallel/vae"

# clip
if [ ! -e "${current_path}/${model_parallel_clip}/clip.om" ]; then
    atc --framework=5 \
        --model=${current_path}/${model_parallel_clip}/clip.onnx \
        --output=${current_path}/${model_parallel_clip}/clip \
        --input_format=ND \
        --log=error \
        --soc_version=Ascend${soc_version}
fi

# unet
if [ ! -e "${current_path}/${model_parallel_unet}/unet.om" ]; then
    atc --framework=5 \
        --model=${current_path}/${model_parallel_unet}/unet_md.onnx \
        --output=${current_path}/${model_parallel_unet}/unet \
        --input_format=NCHW \
        --log=error \
        --optypelist_for_implmode="Gelu,Sigmoid" \
        --op_select_implmode=high_performance \
        --soc_version=Ascend${soc_version}
fi

# vae
if [ ! -e "${current_path}/${model_parallel_vae}/vae.om" ]; then
    atc --framework=5 \
        --model=${current_path}/${model_parallel_vae}/vae.onnx \
        --output=${current_path}/${model_parallel_vae}/vae \
        --input_format=NCHW \
        --log=error \
        --soc_version=Ascend${soc_version}
fi