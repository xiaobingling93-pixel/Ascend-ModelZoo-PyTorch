OMP_NUM_THREADS=1 
torchrun --standalone --nnodes=1 --nproc_per_node=8  ./scripts/finetune_hf.py \
    ./scripts/AdvertiseGen/ \
    ./model \
    ./scripts/configs/sft.yaml \
    True
