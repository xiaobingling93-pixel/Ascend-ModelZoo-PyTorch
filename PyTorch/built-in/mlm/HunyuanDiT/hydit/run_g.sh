model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
deepspeed --num_gpus 8 --num_nodes 1 --master_port=29000 hydit/train_deepspeed.py ${params}  "$@"

#HOSTFILE="/home/l50041210/HunyuanDiT_combine/hostfile"
#MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
#MASTER_PORT=6001
#NODE_ADDR=`hostname -I | awk '{for(i=1;i<=NF;i++)print $i}' | grep ${MASTER_ADDR%.*}. | awk -F " "'{print$1}'`
#NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
#NNODES=$(cat $HOSTFILE | wc -l)
#NPUS_PER_NODE=8
#WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
#echo $MASTER_ADDR
#echo $NODE_ADDR
#echo $NODE_RANK
#echo $NNODES
#
#DISTRIBUTED_ARGS="
#    --nproc_per_node $NPUS_PER_NODE \
#    --nnodes $NNODES \
#    --node_rank $NODE_RANK \
#    --master_addr $MASTER_ADDR \
#    --master_port $MASTER_PORT
##"
#
#torchrun $DISTRIBUTED_ARGS hydit/train_deepspeed.py ${params} "$@"