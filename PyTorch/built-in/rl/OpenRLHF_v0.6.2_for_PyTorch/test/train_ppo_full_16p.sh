#!/bin/bash

echo "-------------------Start PPO Train-------------------"

# 网络名称
RL_ALGO="PPO"

# 默认值
ckpt_path="ppo_ckpts"
tensorboard_path="ppo_tb_log"
output_log="ppo_train_log.log"

# 遍历所有传入的参数
for para in $*; do
  if [[ $para == --dataset_path* ]]; then
    dataset_path="${para#*=}"
  elif [[ $para == --model_path=* ]]; then
    model_path="${para#*=}"
  else
    echo "Unknown parameter: $para" >&2
  fi
done

# 检查参数dataset_path、model_path是否已提供
if [ -z "$dataset_path" ] || [ -z "$model_path" ]; then
  echo "Error: Both dataset_path and model_path are required."
  exit 1
fi

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi


if [ -d ${cur_path}/test/output/${RL_ALGO} ]; then
  rm -rf ${cur_path}/test/output/${RL_ALGO}
  mkdir -p ${cur_path}/test/output/${RL_ALGO}
else
  mkdir -p ${cur_path}/test/output/${RL_ALGO}
fi 

# 启动ray服务
ray stop
source ${test_path_dir}/env_npu.sh
ulimit -n 65535
ray start --head --port 6379


python -m openrlhf.cli.orm_server \
    --dataset $dataset_path \
    --model_name $model_path \
    --log_dir ./logs/openrlhf_train_ppo 2>&1 &

# 训练开始时间
start_time=$(date +%s)

nohup ray job submit --address="http://127.0.0.1:8265" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --pretrain $model_path \
    --save_path ${cur_path}/test/output/${RL_ALGO}/${ckpt_path} \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --temperature 0.6 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 5 \
    --prompt_max_len 1024 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --remote_rm_url http://localhost:8000/get_reward \
    --prompt_data $dataset_path \
    --input_key input \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --use_tensorboard ${cur_path}/test/output/${RL_ALGO}/${tensorboard_path} \
    --save_steps -1 > ${cur_path}/test/output/${RL_ALGO}/${output_log} 2>&1 &

train_program_pid=$!
wait $train_program_pid

# 训练结束时间
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
e2e_time_hours=$(awk 'BEGIN {print "'"$e2e_time"'" / 3600}')
echo "E2E Time (hour) = ${e2e_time_hours}"

# 关键信息打印到${CaseName}.log中，不需要修改
CaseName=${RL_ALGO}_info
echo "RL ALGO = ${RL_ALGO}" > ${cur_path}/test/output/${RL_ALGO}/${CaseName}.log
echo "E2E Time (hour) = ${e2e_time_hours}" >> ${cur_path}/test/output/${RL_ALGO}/${CaseName}.log

#停止orm_server
pkill -f "python -m openrlhf.cli.orm_server"
#停止ray
ray stop

echo "-------------------End PPO Train-------------------"
