source ./test/env_npu.sh
#将Host日志输出到串口,0-关闭/1-开启。指定0关闭日志打屏，即日志采用默认输出方式，将日志保存在log文件中。
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error。此处指定3输出error级别日志，可根据具体需要调整。
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1

#可通过此环境变量配置task_queue算子下发队列是否开启和优化等级。
#-配置为0时：关闭task_queue算子下发队列优化。
#-配置为1或未配置时：开启task_queue算子下发队列Level 1优化。
#-配置为2时：开启task_queue算子下发队列Level 2优化。关于Level 1和Level 2优化的详细解释请查看官网文档。
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export COMBINED_ENABLE=1
export DYNAMIC_COMPILE_ENABLE=0
export EXPERIMENTAL_DYNAMIC_PARTITION=0
#设置应用类日志是否开启Event日志。0-关闭/1-开启，默认值为1，此处设置为0表示关闭Event日志。
export ASCEND_GLOBAL_EVENT_ENABLE=0
#HCCL白名单开关,1-关闭/0-开启。设置为1则无需校验HCCL通信白名单。
export HCCL_WHITELIST_DISABLE=1

export RANK_SIZE=8
KERNEL_NUM=$(($(nproc)/8))

for((RANK_ID=0;RANK_ID<RANK_SIZE;RANK_ID++))
do
    export RANK=$RANK_ID

    if [ $(uname -m) = "aarch64" ]
    then
        PID_START=$((KERNEL_NUM * RANK_ID))
        PID_END=$((PID_START + KERNEL_NUM - 1))
        taskset -c $PID_START-$PID_END \
        ./tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py \
            --launcher pytorch \
            --seed 1 \
            --deterministic \
            --device npu \
            --local_rank 0 &
    else
        python3 ./tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py \
            --launcher pytorch \
            --seed 1 \
            --deterministic \
            --device npu \
            --local_rank 0 &
    fi
done