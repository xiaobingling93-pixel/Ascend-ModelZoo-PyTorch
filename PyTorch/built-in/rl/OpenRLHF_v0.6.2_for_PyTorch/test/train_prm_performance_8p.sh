#!/bin/bash
echo "-------------------Start PRM Train-------------------"

#网络名称
Network="prm"

# 默认值
max_epochs=1
pretrain_path=""
dataset_path=""

# 遍历所有传入的参数
for para in $*; do
  if [[ $para == --max_epochs* ]]; then
    max_epochs="${para#*=}"
  elif [[ $para == --pretrain_path=* ]]; then
    pretrain_path="${para#*=}"
  elif [[ $para == --dataset_path=* ]]; then
    dataset_path="${para#*=}"
  else
    echo "Unknown parameter: $para" >&2
  fi
done

# 检查参数pretrain_path、dataset_path是否已提供
if [ -z "$pretrain_path" ] || [ -z "$dataset_path" ]; then
  echo "Error: Both pretrain_path and dataset_path are required."
  exit 1
fi

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

#创建KTO训练输出目录，不需要修改
if [ -d ${cur_path}/test/output/${Network} ]; then
  rm -rf ${cur_path}/test/output/${Network}
  mkdir -p ${cur_path}/test/output/${Network}
else
  mkdir -p ${cur_path}/test/output/${Network}
fi 

source ${test_path_dir}/env_npu.sh
#可通过此环境变量配置task_queue算子下发队列是否开启和优化等级。
#-配置为0时：关闭task_queue算子下发队列优化。
#-配置为1或未配置时：开启task_queue算子下发队列Level 1优化。
#-配置为2时：开启task_queue算子下发队列Level 2优化。关于Level 1和Level 2优化的详细解释请查看官网文档。
export TASK_QUEUE_ENABLE=2

#设置是否开启combined标志,0-关闭/1-开启。设置为1表示开启，用于优化非连续两个算子组合类场景。
export COMBINED_ENABLE=1

#缓存算子信息条目数
export ACLNN_CACHE_LIMIT=100000

read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --save_path ./checkpoint/mistal-7b-prm \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 64 \
   --micro_train_batch_size 8 \
   --max_samples 64000 \   
   --pretrain $pretrain_path \
   --bf16 \
   --max_epochs $max_epochs \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --dataset $dataset_path \
   --input_key input \
   --label_key value \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token ки \
   --reward_tokens + -
EOF

if [[ ${1} != "slurm" ]]; then
  deepspeed --module $training_commands > ${cur_path}/test/output/${Network}/train_${Network}.log 2>&1 &
fi

wait

# 训练用例信息，不需要修改
DeviceType=$(uname -m)
CaseName=${Network}_info

# 获取训练日志
source_log_file="${cur_path}/test/output/${Network}/train_${Network}.log"
target_log_file="${cur_path}/test/output/${Network}/train_${Network}_loss.log"

max_lines=1000

grep -oP "{'train/prm_loss':.*}" "$source_log_file" > "$target_log_file"

#计算平均单步耗时，默认1000步（去除前100步）
tps=$(grep -a "'train/prm_loss':" "$target_log_file" |
  awk -F "'train/step_time': '" '{print $2}' |
  awk -F "'" '{print $1}' |
  awk -v max_lines="$max_lines" '{gsub(/s/, ""); if (NR > 100 && NR <= max_lines) {sum+=$1; count++}} END {if (count>0) print sum/count; else print "No data"}')
echo "Second Per Step: $tps"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${cur_path}/test/output/${Network}/${CaseName}.log
echo "Second Per Step = ${tps}" > ${cur_path}/test/output/${Network}/${CaseName}.log
echo "-------------------End PRM Train-------------------"