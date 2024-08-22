source /home/l50041210/cann-b20/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3

python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚" --image-size 1280 768 --no-enhance