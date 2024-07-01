#!/bin/bash
source groundingdino_npu/env_npu.sh
PYTHON_PATH="Python Env Path"
export Mx_Driving_PYTHON_PATH=${PYTHON_PATH}/lib/python3.8
export ASCEND_CUSTOM_OPP_PATH=${Mx_Driving_PYTHON_PATH}/site-packages/mx_driving/packages/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/:$LD_LIBRARY_PATH

python groundingdino_npu/image_demo_npu.py \
	demo/demo.jpg \
	groundingdino_npu/mm_grounding_dino_swin-b_inference.py \
	--weights "../weights/grounding_dino_swin-b_pretrain_all-f9818a7c.pth" \
	--texts 'bench . car .'

