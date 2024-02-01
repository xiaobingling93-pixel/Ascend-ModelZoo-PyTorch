# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import deepspeed
from flagai.auto_model.auto_loader import AutoLoader
from optim_factory import create_optimizer, LayerDecayValueAssigner, get_parameter_groups, get_is_head_flag_for_vit
import os 
from flagai.trainer import Trainer
from torchvision.datasets import (
    CIFAR10
)
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_device_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_device_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_root = "./clip_benchmark_datasets"
dataset_name = "cifar10"

batch_size = 4
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

auto_loader = AutoLoader(
    task_name="txt_img_matching",
    model_dir="./checkpoints",
    model_name="AltCLIP-XLMR-L"   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
)

model = auto_loader.get_model()
model.to(device)
model.eval()
tokenizer = auto_loader.get_tokenizer()
transform = auto_loader.get_transform()

model_hidden_size = 1024
train_batch_size = 2

ds_config = {
        "optimizer": {
            "type": "Adam",
            "adam_w_mode": True,
            "params": {
              "lr": 2e-05,
              "weight_decay": 0.05,
              "bias_correction": True,
              "betas": [
                0.9,
                0.98
              ],
              "eps": 1e-06
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": model_hidden_size * model_hidden_size,
            "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
            "stage3_param_persistence_threshold": 10 * model_hidden_size
        },
        "steps_per_print": 2000,
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False
}

## use deep-speed
num_layers = model.vision_model.encoder.layers.__len__()
layer_decay = 0.9
weight_decay = 0.05
task_head_lr_weight = 1.0
if layer_decay < 1.0:
    lrs = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
    assigner = LayerDecayValueAssigner(lrs)
elif task_head_lr_weight > 1:
    assigner = LayerDecayValueAssigner([1.0, task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
else:
    assigner = None

if assigner is not None:
    print("Assigned values = %s" % str(assigner.values))
skip_weight_decay_list = []
optimizer_params = get_parameter_groups(
    model, weight_decay, skip_weight_decay_list,
    assigner.get_layer_id if assigner is not None else None,
    assigner.get_scale if assigner is not None else None)
model, optimizer, _, _ = deepspeed.initialize(config_params=ds_config,
        model=model, model_parameters=optimizer_params,
        dist_init_required=False,
    )

trainer = Trainer(env_type="pytorch",
                pytorch_device=device,
                experiment_name="clip_finetuning",
                batch_size=4,
                lr=1e-4,
                epochs=10,
                log_interval=10)

dataset = CIFAR10(root=os.path.join(dataset_root, dataset_name), 
                transform=transform,   
                download=True)

def cifar10_collate_fn(batch):
    # image shape is (batch, 3, 224, 224)
    images = torch.tensor([b[0]["pixel_values"][0] for b in batch])
    # text_id shape is (batch, n)
    input_ids = torch.tensor([tokenizer(f"a photo of a {b[1]}",
                                padding=True,
                                truncation=True,
                                max_length=77)["input_ids"] for b in batch])    

    attention_mask = torch.tensor([tokenizer(f"a photo of a {b[1]}",
                                padding=True,
                                truncation=True,
                                max_length=77)["attention_mask"] for b in batch])

    return {
        "pixel_values": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
if __name__ == "__main__":
    trainer.train(model=model, train_dataset=dataset, collate_fn=cifar10_collate_fn)