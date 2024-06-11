#!/bin/bash

#当前路径,不需要修改
DIR=`pwd`

##################基础参数，需要模型审视修改###################
# 网络名称，训练资源准备
Network="Qwen-VL"
model_name="Qwen/Qwen-VL-Chat"
data_path="path_to_data"
# 模型训练参数
batch_size=1
gradient_accumulation_steps=16
model_max_length=2048
epochs=5
output_path="./output-qwen-vl"

#######################机器环境配置#########################
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001


# 参数传入
for para in $*
do
    if [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --model_name* ]];then
        model_name=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --gradient_accumulation_steps* ]];then
        gradient_accumulation_steps=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --output_path* ]];then
        output_path=`echo ${para#*=}`
    fi
done


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
DIR=`pwd`
cur_path_last_diename=${DIR##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${DIR}
    cd ..
    DIR=`pwd`
else
    test_path_dir=${DIR}/test
fi

#################创建日志输出目录，不需要修改#################
ASCEND_DEVICE_ID="8p"
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
  source ${test_path_dir}/env_npu.sh
fi

#################启动训练脚本#################
start_time=$(date +%s)

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $model_name \
    --data_path $data_path \
    --bf16 True \
    --fix_vit True \
    --output_dir $output_path \
    --num_train_epochs $epochs \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length $model_max_length \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8 \
    --deepspeed finetune/ds_config_zero2.json > ${test_path_dir}/output/8p/train_full_8p.log 2>&1 &

wait

end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

##################获取训练数据################
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a 's/it' ${test_path_dir}/output/8p/train_full_8p.log|awk -F ', ' '{print $2}'|awk -F 's/it' '{print $1}'|tail -n 10|awk '{sum+=$1} END {print sum/NR}'`
ActualFPS=$(echo "scale=2; $model_max_length*$batch_size*$gradient_accumulation_steps/$FPS" | bc)

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_'full'

#获取性能数据，不需要修改
#从train_full_8p.log提取Loss到train_${CaseName}_loss.txt中，需要模型审视修改
grep -a "'loss':" ${test_path_dir}/output/8p/train_full_8p.log|awk -F ': ' '{print $2}' > ${test_path_dir}/output/8p/train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/8p/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
