#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="OpenFold_for_PyTorch"

cur_path=`pwd`
max_epochs=1

for para in $*
do
   if [[ $para == --max_epochs* ]];then
      	max_epochs=`echo ${para#*=}`
   elif [[ $para == --data_path* ]];then
       data_path=`echo ${para#*=}`
   fi
done

source ${cur_path}/test/env_npu.sh

python3 train_openfold.py  $data_path/pdb_data/mmcif_files \
                           $data_path/alignment_data/alignment_dbs \
                           $data_path/pdb_data/mmcif_files \
                           ./output \
                           "2021-10-10" \
                           --train_chain_data_cache_path $data_path/pdb_data/chain_data_cache.json \
                           --template_release_dates_cache_path $data_path/pdb_data/mmcif_cache.json \ 
                           --alignment_index_path $data_path/alignment_data/alignment_dbs/alignment_db.index \
                           --config_preset initial_training \
                           --seed 42 \
                           --obsolete_pdbs_file_path $data_path/pdb_data/obsolete.dat \
                           --num_nodes 1 \
                           --gpus 8 \
                           --max_epochs $max_epochs \
                           --deepspeed_config_path ./deepspeed_config.json \
                           --checkpoint_every_epoch \
                           --openfold_run_stage train \
                           > openfold_train_8p.log 2>&1 &

wait

# 取最后一步的打屏时间
read a b c <<< $(tac openfold_train_8p.log | grep -m1 '1250/1250' | sed -n 's/.*\[\([0-9]\+\):\([0-9]\+\):\([0-9]\+\).*/\1 \2 \3/p')

# 计算总秒数
total_seconds=$((a*3600 + b*60 + c))

#打印，不需要修改
echo "E2E Total Time sec : $total_seconds"
echo "Format time: $a:$b:$c"