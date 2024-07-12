OMP_NUM_THREADS=1 

save_log=run_train_8p_glm_no_shuffle.log

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

torchrun --standalone --nnodes=1 --nproc_per_node=8  ./scripts/finetune_hf.py \
    ./scripts/AdvertiseGen/ \
    ./model \
    ./scripts/configs/sft.yaml \
    False >$save_log 2>&1 &

wait
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))


#结果打印，不需要修改
echo "------------------ Final result ------------------"

#获取性能数据，不需要修改
ActualPerformance=`cat $save_log | grep -a "train_samples_per_second" | awk -F "train_samples_per_second" '{print $NF}' | cut -d "," -f 1 | cut -d " " -f 2`
echo "Final Performance train_samples/sec : $ActualPerformance"

#loss值，不需要修改
ActualLoss=$(grep -o "'loss': [0-9.]*" $save_log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"