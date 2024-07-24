#!/bin/bash
source scripts/npu_set_env.sh
currentDir=`pwd`
echo "test log path is ${currentDir}/test.log"
train_epoch=300
python3 -u test.py -m weights/FaceBoxes_epoch_$train_epoch.pth --cpu --dataset FDDB --save_folder eval_$train_epoch > test.log 2>&1
