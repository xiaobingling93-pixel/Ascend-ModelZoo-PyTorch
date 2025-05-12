#!/bin/bash
echo "-------------------Start SFT Train-------------------"

#配置八卡
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

#网络名称
Network="sft"

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

#创建SFT训练输出目录，不需要修改
if [ -d ${cur_path}/test/output/${Network} ]; then
  rm -rf ${cur_path}/test/output/${Network}
  mkdir -p ${cur_path}/test/output/${Network}
else
  mkdir -p ${cur_path}/test/output/${Network}
fi 

source ${test_path_dir}/env_npu.sh

training_commands=$(cat <<EOF
openrlhf.cli.train_vl_sft \
   --max_len 2048 \
   --dataset $dataset_path \
   --dataset_config_path $cur_path/examples/vision_scripts/llava_zh_300k.json \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --max_samples 6000 \
   --processing_num_workers 8 \
   --overwrite_cache True \
   --model_arch qwen2_vl \
   --pretrain $pretrain_path \
   --save_path $cur_path/checkpoint/qwen2vl_sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs $max_epochs \
   --bf16 \
   --flash_attn sdpa \
   --learning_rate 1e-5 \
   --lr_scheduler constant
EOF
)

if [[ ${1} != "slurm" ]]; then
  deepspeed --master_port=12432 --module $training_commands > ${cur_path}/test/output/${Network}/train_${Network}.log 2>&1 &
fi

wait

# 训练用例信息，不需要修改
DeviceType=$(uname -m)
CaseName=${Network}_info

# 获取训练日志
source_log_file="${cur_path}/test/output/${Network}/train_${Network}.log"

#计算全程平均单步耗时
tps=$(grep -a "'train/step_time':" "$source_log_file" |
  sed 's/\x1b\[[0-9;]*m//g' |
  awk -F "'train/step_time': '" '{print $2}' |
  awk -F "'," '{print $1}' |
  awk '{gsub(/s/, ""); sum+=$1; count++} END {if (count>0) print sum/count; else print "No data"}')
echo "Second Per Step: $tps"

# 关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${cur_path}/test/output/${Network}/${CaseName}.log
echo "Second Per Step = ${tps}" > ${cur_path}/test/output/${Network}/${CaseName}.log
echo "-------------------End SFT Train-------------------"