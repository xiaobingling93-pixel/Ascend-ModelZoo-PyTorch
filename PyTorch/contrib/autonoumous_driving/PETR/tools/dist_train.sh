# ------------------------------------------------------------------------
# Copyright 2024 Huawei Technologies Co., Ltd
#-------------------------------------------------------------------------

#!/usr/bin/env bash

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
CONFIG=projects/configs/petr/petr_r50dcn_gridmask_p4.py
GPUS=8
PORT=${PORT:-28500}

echo "sigle node train. npu_num:${GPUS}"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --use_env --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
