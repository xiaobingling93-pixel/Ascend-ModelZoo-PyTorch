#!/bin/bash
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'

if [ -f $CANN_INSTALL_PATH_CONF ]; then
    CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
    CANN_INSTALL_PATH="/usr/local/Ascend"
fi

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ]; then
    source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
    source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi


# 控制对输入数据为Inf/NaN的处理能力
# 0：饱和模式，计算出现溢出时，计算结果会饱和为浮点数极值（+-MAX）。
# 1：INf_NAN模式，根据定义输出Inf/NaN的计算结果。
# Atlas训练系列仅支持饱和模式，Atlas A2/A3默认值为1，支持配置为0的饱和模式。
export INF_NAN_MODE_ENABLE=0
# Atlas A2/A3训练系列产品，INF_NAN_MODE_ENABLE默认为“1”INF_NAN模式。如模型中使用了Inf/NaN，配置为“0”饱和模式时，会有不可预期的精度问题。
# 若要强制配置为“0”饱和模式，则会被拦截报错，若一定要关闭INF_NAN模式开启饱和模式，需配置INF_NAN_MODE_FORCE_DISABLE=1。
export INF_NAN_MODE_FORCE_DISABLE=1

#将Host日志输出到串口,0-关闭/1-开启。指定0关闭日志打屏，即日志采用默认输出方式，将日志保存在log文件中。
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error。此处指定3输出error级别日志，可根据具体需要调整。
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置应用类日志是否开启Event日志。0-关闭/1-开启，默认值为1，此处设置为0表示关闭Event日志。
export ASCEND_GLOBAL_EVENT_ENABLE=0

#可通过此环境变量配置task_queue算子下发队列是否开启和优化等级。
#-配置为0时：关闭task_queue算子下发队列优化。
#-配置为1或未配置时：开启task_queue算子下发队列Level 1优化。
#-配置为2时：开启task_queue算子下发队列Level 2优化。关于Level 1和Level 2优化的详细解释请查看官网文档。
export TASK_QUEUE_ENABLE=0

#设置是否开启PTCopy,0-关闭/1-开启
export PTCOPY_ENABLE=1
#设置是否开启combined标志,0-关闭/1-开启。设置为1表示开启，用于优化非连续两个算子组合类场景。
export COMBINED_ENABLE=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD"
#HCCL白名单开关,1-关闭/0-开启。设置为1则无需校验HCCL通信白名单。
export HCCL_WHITELIST_DISABLE=1
#设置Device侧日志等级为error
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7
#关闭Device侧Event日志
msnpureport -e disable


