# AnimateDiff for PyTorch



# 目录

- [AnimateDiff](#animatediff-for-pytorch)
  - [概述](#概述)
  - [准备训练环境](#准备训练环境)
    - [创建Python环境](#创建Python环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
    - [准备推理权重](#准备推理权重)
  - [快速开始](#快速开始)
    - [模型训练](#模型训练)
    - [训练结果](#训练结果)
    - [模型推理](#模型推理)
  - [公网地址说明](#公网地址说明)
  - [变更说明](#变更说明)
  - [FAQ](#faq)



## 概述

### 模型介绍

AnimateDiff提出了一个有效的框架，可将现有的大多数个性化文本到图像模型一次性制成动画，从而节省了针对特定模型进行微调的工作量。

本仓已经支持以下模型任务类型

|     模型      | 任务列表 | 是否支持 |
|:-----------:|:----:|:-----:|
| AnimateDiff |  微调  | ✔ |
| AnimateDiff |  推理  | ✔ |

- 参考实现：

  ```
  url=https://github.com/guoyww/AnimateDiff.git
  commit_id=cf80ddeb47b69cf0b16f225800de081d486d7f21
  ```

- 适配昇腾AI处理器的实现：
  ```shell
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/AnimateDiff
  ```

## 准备训练环境

### 创建Python环境

- git clone 远程仓
  ```shell
    git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
    cd PyTorch/built-in/mlm/AnimateDiff
  ```

- 创建Python环境并且安装Python三方包
  ```shell
    conda env create -f environment.yaml
    conda activate animatediff
    pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu  # For X86
    pip3 install torch==2.1.0  # For Aarch64
    pip3 install accelerate==0.28.0 diffusers==0.11.1 decorator==5.1.1 scipy==1.12.0 attrs==23.2.0  torchvision==0.16.0 transformers==4.25.1 huggingface_hub==0.23.2
  ```
- 环境准备指导

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

    **表 1**  昇腾软件版本支持表

  |     软件类型     |   支持版本   |
  |:-----------:|:--------:|
  | FrameworkPTAdapter  |   在研版本   |
  | CANN | 在研版本 |
   | 昇腾NPU固件 | 在研版本 |
   | 昇腾NPU驱动 | 在研版本 |


### 准备数据集

- 需要自行下载WebVid10M数据集，下载并整理的数据集如下所示：
   ```
    数据集结构
    ├── 2M_val
    │   ├── 10003109.mp4
    │   ├── 10023815.mp4
    │   ├── 10024310.mp4
    │   ├── 10042700.mp4
    │   ├── 10052036.mp4
    │   ├── 10052783.mp4
    │   ├── 1005608956.mp4
    └── results_2M_val.csv
   ```
  数据来源可以参考 https://github.com/guoyww/AnimateDiff/blob/main/__assets__/docs/animatediff.md 中的数据准备章节。
### 准备预训练权重

- 需要准备2个模型权重:
  ```shell
  runwayml/stable-diffusion-v1-5
  openai/clip-vit-large-patch14
  ```
- 将stable-diffusion-v1-5 路径传入到configs/training/v1/image_finetune.yaml 的pretrained_model_path。
- openai/clip-vit-large-patch14需要放置到模型的根目录下面。
### 准备推理权重
- 如果想使用AnimateDiff的推理功能需要下载下面提及到的模型，按照模型的名字对应放到models目录下面DreamBooth_LoRA、MotionLoRA、Motion_Module、SparseCtrl、StableDiffusion文件夹中。
 模型获取可以参考 https://github.com/guoyww/AnimateDiff/blob/main/__assets__/docs/animatediff.md 文档。
  <details>
  <summary>点击展开模型详情</summary>
  
  ```shell
  models
    ├── DreamBooth_LoRA
    │   ├── CounterfeitV30_v30.safetensors
    │   ├── FilmVelvia2.safetensors
    │   ├── lora_Ghibli_n3.safetensors
    │   ├── lyriel_v16.safetensors
    │   ├── majicmixRealistic_v4.safetensors
    │   ├── majicmixRealistic_v5Preview.safetensors
    │   ├── moonfilm_reality20.safetensors
    │   ├── Put personalized T2I checkpoints here.txt
    │   ├── rcnzCartoon3d_v10.safetensors
    │   ├── realisticVisionV20_v20.safetensors
    │   ├── realisticVisionV40_v20Novae.safetensors
    │   ├── realisticVisionV51_v51VAE.safetensors
    │   ├── toonyou_beta3.safetensors
    │   ├── toonyou_beta6.safetensors
    │   └── TUSUN.safetensors
    ├── MotionLoRA
    │   ├── Put MotionLoRA checkpoints here.txt
    │   ├── v2_lora_PanLeft.ckpt
    │   ├── v2_lora_PanRight.ckpt
    │   ├── v2_lora_RollingAnticlockwise.ckpt
    │   ├── v2_lora_RollingClockwise.ckpt
    │   ├── v2_lora_TiltDown.ckpt
    │   ├── v2_lora_TiltUp.ckpt
    │   ├── v2_lora_ZoomIn.ckpt
    │   └── v2_lora_ZoomOut.ckpt
    ├── Motion_Module
    │   ├── mm_sd_v14.ckpt
    │   ├── mm_sd_v15.ckpt
    │   ├── mm_sd_v15_v2.ckpt
    │   ├── Put motion module checkpoints here.txt
    │   ├── v3_sd15_adapter.ckpt
    │   └── v3_sd15_mm.ckpt
    ├── SparseCtrl
    │   ├── v3_sd15_sparsectrl_rgb.ckpt
    │   └── v3_sd15_sparsectrl_scribble.ckpt
    └── StableDiffusion
        ├── Put diffusers stable-diffusion-v1-5 repo here.txt
        └── stable-diffusion-v1-5
            ├── feature_extractor
            │   └── preprocessor_config.json
            ├── model_index.json
            ├── README.md
            ├── safety_checker
            │   ├── config.json
            │   └── pytorch_model.bin
            ├── scheduler
            │   └── scheduler_config.json
            ├── text_encoder
            │   ├── config.json
            │   └── pytorch_model.bin
            ├── tokenizer
            │   ├── merges.txt
            │   ├── special_tokens_map.json
            │   ├── tokenizer_config.json
            │   └── vocab.json
            ├── unet
            │   ├── config.json
            │   └── diffusion_pytorch_model.bin
            ├── v1-5-pruned.ckpt
            ├── v1-5-pruned-emaonly.ckpt
            ├── v1-inference.yaml
            └── vae
                ├── config.json
                └── diffusion_pytorch_model.bin
  ```
  </details>

## 快速开始

### 模型训练

1. 对configs/training/v1/image_finetune.yaml进行修改，需要将数据集路径、训练步数、验证步数传递到yaml中。要修改的点如下所示：
   ```shell
    pretrained_model_path: "./stable-diffusion-v1-5/" # 修改为stable-diffusion-v1-5模型的路径
    train_data:
      csv_path:    "./results_2M_val.csv" # 修改为上述csv数据集的路径
      video_folder: "./2M_val" # 修改为上述2M_val数据集的路径
      sample_size:   256
    train_batch_size: 64 # 训练batchsize
    max_train_steps:  2000 # 最大训练步数
    checkpointing_steps:  2000 # 每多少步保存一次ckpt
    validation_steps: 2000 # 每多少步做一次验证
    validation_steps_tuple: [] # 每多少步做一次验证
    enable_xformers_memory_efficient_attention: false # npu不支持xformers
   ```

2. 运行下面训练脚本，该模型支持单机8卡训练。

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    torchrun --nnodes=1 --nproc_per_node=8 --master_port 29511  train.py --config configs/training/v1/image_finetune.yaml
    ```
### 训练结果

**表 2**  训练结果展示表

| 芯片 | 卡数 | samples per second | batch_size | AMP_Type | Torch_Version |
|:---:|:---:|:------------------:|:----------:|:--------:|:---:|
| GPU | 8p |       469.1        |     64     |   fp16   | 2.1 |
| Atlas A2 | 8p |       410.7        |     64     |   fp16   | 2.1 |

### 模型推理

  执行下面的shell脚本，进行推理。
   ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python -m scripts.animate --config configs/prompts/v3/v3-1-T2V.yaml
    python -m scripts.animate --config configs/prompts/v3/v3-2-animation-RealisticVision.yaml
    python -m scripts.animate --config configs/prompts/v3/v3-3-sketch-RealisticVision.yaml
    python -m scripts.animate --config configs/prompts/v2/v2-1-RealisticVision.yaml
    python -m scripts.animate --config configs/prompts/v2/v2-2-RealisticVision-MotionLoRA.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-1-ToonYou.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-2-Lyriel.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-3-RcnzCartoon.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-4-MajicMix.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-5-RealisticVision.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-6-Tusun.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-7-FilmVelvia.yaml
    python -m scripts.animate --config configs/prompts/v1/v1-8-GhibliBackground.yaml
   ```

## 公网地址说明
代码涉及公网地址参考 public_address_statement.md

## 变更说明
2024.05.11: 首次发布

2024.05.20: 加入npu适配代码

## FAQ
Q: 为什么train.py中的num_workers改为0？

A: 在X86机器（包括GPU和NPU）上进行多卡+多进程处理数据时，多进程并发处理数据会卡主。源仓代码是单卡微调，则不会出现这个问题。如果你使用的是ARM的机器则可以使用多进程处理数据。
