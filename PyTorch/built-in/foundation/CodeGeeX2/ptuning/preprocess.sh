source env_npu.sh
python preprocess.py \
    --do_train \
    --train_file train.json \
    --test_file dev.json \
    --prompt_column PROMPT \
    --response_column ANSWER \
    --model_name_or_path /home/h00638954/codegeex2-6b/ \
    --overwrite_cache \
    --output_dir ./output/adgen-chatglm-6b-ft-$LR \
    --max_source_length 2048 \
    --max_target_length 2048
:<<COMMOT
python preprocess.py \
    --do_predict \
    --train_file train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /home/h00638954/codegeex2-6b/ \
    --overwrite_cache \
    --output_dir ./output/adgen-chatglm-6b-ft-$LR \
    --max_source_length 4096 \
    --max_target_length 4096
COMMOT
