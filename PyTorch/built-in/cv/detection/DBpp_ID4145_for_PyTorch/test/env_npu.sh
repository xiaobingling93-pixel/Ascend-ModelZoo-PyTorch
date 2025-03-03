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


#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0

#可通过此环境变量配置task_queue算子下发队列是否开启和优化等级。
#-配置为0时：关闭task_queue算子下发队列优化。
#-配置为1或未配置时：开启task_queue算子下发队列Level 1优化。
#-配置为2时：开启task_queue算子下发队列Level 2优化。关于Level 1和Level 2优化的详细解释请查看官网文档。
export TASK_QUEUE_ENABLE=0
#设置是否开启2个非连续combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#HCCL默认超时时间120s较少，修改为1800s对齐PyTorch默认设置
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1200
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')

ulimit -SHn 512000

#设置device侧日志登记为error
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

