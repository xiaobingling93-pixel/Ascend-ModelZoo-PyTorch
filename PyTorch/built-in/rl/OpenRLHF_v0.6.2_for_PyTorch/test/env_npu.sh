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
msnpureport -g error -d 0
msnpureport -g error -d 1
msnpureport -g error -d 2
msnpureport -g error -d 3
msnpureport -g error -d 4
msnpureport -g error -d 5
msnpureport -g error -d 6
msnpureport -g error -d 7

# 控制对输入数据为Inf/NaN的处理能力
# 0：饱和模式，计算出现溢出时，计算结果会饱和为浮点数极值（+-MAX）。
# 1：INf_NAN模式，根据定义输出Inf/NaN的计算结果。
# Atlas训练系列仅支持饱和模式，Atlas A2/A3默认值为1，支持配置为0的饱和模式。
export INF_NAN_MODE_ENABLE=1
#将Host日志输出到串口,0-关闭/1-开启。指定0关闭日志打屏，即日志采用默认输出方式，将日志保存在log文件中。
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error。此处指定3输出error级别日志，可根据具体需要调整。
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置应用类日志是否开启Event日志。0-关闭/1-开启，默认值为1，此处设置为0表示关闭Event日志。
export ASCEND_GLOBAL_EVENT_ENABLE=0

#HCCL白名单开关,1-关闭/0-开启。设置为1则无需校验HCCL通信白名单。
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')