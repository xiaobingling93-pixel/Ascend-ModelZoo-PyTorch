
#LR=1e-4
LR=2e-5

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
source env_npu.sh
#deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
deepspeed --num_gpus=8 --master_port $MASTER_PORT main_without_tokenizer.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file train.json \
    --test_file dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /home/h00638954/codegeex2-6b/ \
    --output_dir ./output/adgen-chatglm2-6b-ft-$LR \
    --overwrite_output_dir \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate $LR \
    --fp16

