model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
deepspeed --num_gpus 8 --num_nodes 1 --master_port=29000 hydit/train_deepspeed.py ${params}  "$@"
