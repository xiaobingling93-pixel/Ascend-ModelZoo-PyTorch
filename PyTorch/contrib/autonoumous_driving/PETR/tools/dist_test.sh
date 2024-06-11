# ------------------------------------------------------------------------
# Copyright 2024 Huawei Technologies Co., Ltd
#-------------------------------------------------------------------------

#!/usr/bin/env bash

CONFIG=projects/configs/petr/petr_r50dcn_gridmask_p4.py
CHECKPOINT=work_dirs/petr_r50dcn_gridmask_p4/latest.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --use_env --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval bbox
