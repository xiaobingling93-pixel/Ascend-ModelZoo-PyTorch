# 目录

- [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [SDXL](#sdxl)
  - [准备训练环境](#准备训练环境)
    - [安装模型环境](#安装模型环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
      - [微调数据集准备](#微调数据集准备)
        - [LoRA微调](#lora微调)
        - [Controlnet微调](#controlnet微调)
        - [全参微调](#全参微调)
    - [获取预训练模型](#获取预训练模型)
  - [快速开始](#快速开始)
    - [微调任务](#微调任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
        - [性能](#性能)
    - [推理任务](#推理任务)
      - [开始推理](#开始推理)
- [SDXL\_Turbo](#sdxl_turbo)
  - [准备训练环境](#准备训练环境-1)
    - [安装模型环境](#安装模型环境-1)
    - [安装昇腾环境](#安装昇腾环境-1)
    - [获取预训练模型](#获取预训练模型-1)
  - [快速开始](#快速开始-1)
    - [推理任务](#推理任务-1)
      - [开始推理](#开始推理-1)
- [AnimateDiff](#animatediff)
  - [准备训练环境](#准备训练环境-2)
    - [安装模型环境](#安装模型环境-2)
    - [安装昇腾环境](#安装昇腾环境-2)
    - [获取预训练模型](#获取预训练模型-2)
    - [获取推理需要的数据](#获取推理需要的数据)
  - [快速开始](#快速开始-2)
    - [推理任务](#推理任务-2)
      - [开始推理](#开始推理-2)
- [DiT](#dit)
  - [准备环境](#准备环境)
    - [安装模型环境](#安装模型环境-3)
    - [安装昇腾环境](#安装昇腾环境-3)
  - [快速开始](#快速开始-3)
    - [推理任务](#推理任务-3)
      - [获取预训练模型](#获取预训练模型-3)
      - [开始推理](#开始推理-3)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
  - [变更](#变更)
- [FAQ](#faq)

# 简介
## diffusers介绍

扩散模型 (Diffusion Models) 是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是Huggingface发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频甚至分子的3D结构。套件包含基于扩散模型的多种个模型，提供了各种下游任务的训练与推理的实现。
本仓库主要将SDXL、SDXL_Turbo、Animatediff、DiT等模型的多个任务迁移到了昇腾NPU上，并进行极致性能优化。

## openmind-diffusers介绍
openmind-diffusers 为 diffusers 的插件仓，通过复用 diffusers 的 plugin 机制，增加其对于昇腾NPU的支持。 其打包的 whl 包名为 diffusers_npu。

## 环境准备

python3.8版本及以上

### 1. 安装依赖

请安装最新昇腾软件栈：https://www.hiascend.com/zh/

| 依赖软件            |
|-----------------|
| Driver          |
| Firmware        |
| CANN            |
| Kernel          |
| PyTorch         |
| torch_npu       |

### 2. 安装 diffusers原生开源库

请参考huggingface Diffusers官方网站的安装指导:
https://huggingface.co/docs/diffusers/installation

### 3. 安装 diffusers_npu
复制测试脚本:
将openmind-diffuser仓库中/examples目录下的所有内容复制到环境中diffusers安装路径对应的examples目录下。


## 支持任务列表

本仓已经支持以下模型任务类型

|     模型      |    任务列表    | 是否支持 |
|:-----------:|:----------:|:-----:|
|    SDXL     |    Lora    | ✔ |
|    SDXL     | Controlnet | ✔ |
|    SDXL     |   全参微调    | ✔ |
|    SDXL     |   文生图推理    | ✔ |
|    SDXL     |   图生图推理    | ✔ |
| SDXL_Turbo  |   文生图推理    | ✔ |
| AnimateDiff |   文生图推理    | ✔ |
| DiT         |   推理         | ✔ |


## 代码实现

- 参考实现：

  ```
  url=https://github.com/huggingface/diffusers
  commit_id=7f551e29ff4ad05615cb38530a8940811f9e5936
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/openmind-ai/openmind-diffusers.git
  code_path=/
  ```


# SDXL

## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  | 三方库    | 支持版本  |
  | :--------: | :-------------: |
  | PyTorch | 2.1.0 |
  | diffusers | 0.28.0 |
  | accelerate | 0.29.3 | 
  | deepspeed | 0.14.1|

   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```shell
   pip install -e .                                                # 安装本地diffusers代码仓
   pip install -r examples/text_to_image/requirements_sdxl.txt     # 安装对应依赖
   pip install -r examples/controlnet/requirements_sdxl.txt
   ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

  | 软件类型   | 支持版本  |
  | :--------: | :-------------: |
  | FrameworkPTAdapter | 在研版本 |
  | CANN | 在研版本 |
  | 昇腾NPU固件 | 在研版本 | 
  | 昇腾NPU驱动 | 在研版本 |

  

### 准备数据集

#### 微调数据集准备
##### LoRA微调

   1. 联网情况下，数据集会自动下载。
   2. 无网络情况下，用户需自行获取pokemon-blip-captions数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径。

   ```shell
   examples/SDXL/sdxl_text2img_lora_deepspeed.sh
   ```

   pokemon-blip-captions数据集格式如下:
   ```
   pokemon-blip-captions
   ├── dataset_infos.json
   ├── README.MD
   └── data
        ├── dataset_infos.json
        └── train-001.parquet
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
   
##### Controlnet微调

   1. 联网情况下，数据集会自动下载。
   2. 无网络情况下，用户需自行获取fill50k数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径，以及需要修改里面fill50k.py文件。

   ```shell
   examples/SDXL/sdxl_text2img_controlnet_deepspeed.sh
   ```
   > **注意：** 
   >需要修改数据集下面的fill50k.py文件中的57到59行，修改示例如下:
   > ```python
   > metadata_path = "数据集路径/fill50k/train.jsonl"
   > images_dir = "数据集路径/fill50k"
   > conditioning_images_dir = "数据集路径/fill50k"
   >```
   fill50k数据集格式如下:
   ```
   fill50k
   ├── images
   ├── conditioning_images
   ├── train.jsonl
   └── fill50k.py
   ```


   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

##### 全参微调 

   > **说明：** 
   >数据集同Lora微调，请参考Lora章节。
 

### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   stabilityai/stable-diffusion-xl-base-1.0 #预训练模型
   madebyollin/sdxl-vae-fp16-fix #vae模型
   ```

3. 获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径。
   ```shell
   examples/SDXL/sdxl_text2img_finetune_deepspeed.sh
   ```
## 快速开始


### 微调任务
本任务主要提供LoRA、Controlnet、全参微调三种微调下游任务的8卡训练脚本（使用deepspeed分布式训练）。

#### 开始训练
   

1. 进入解压后的源码包根目录。

    ```
   cd /${模型文件夹名称} 
   ```


2. 运行训练的脚本。
- 单机八卡微调
  ```shell
  bash examples/SDXL/sdxl_text2img_controlnet_deepspeed.sh      #8卡deepspeed训练 sdxl_controlnet fp16
  bash examples/SDXL/sdxl_text2img_lora_deepspeed.sh            #8卡deepspeed训练 sdxl_lora fp16
  bash examples/SDXL/sdxl_text2img_finetune_deepspeed.sh        #8卡deepspeed训练 sdxl_finetune fp16
  ```
 - 微调脚本参数说明如下
 ```shell
  examples/text_to_image/train_text_to_image_lora_sdxl.py or examples/text_to_image/train_controlnet_sdxl.py or examples/text_to_image/train_text_to_image_sdxl.py
  --pretrained_model_name_or_path    //基础模型路径
  --dataset_name                     //数据集名称
  --resolution                       //分辨率大小
  --train_batch_size                 //训练batchsize
  --num_train_epochs                 //训练epochs次数
  --checkpointing_steps              //每steps保存一次
  --learning_rate                    //学习率
  --lr_scheduler                     //学习率衰减策略
  --lr_warmup_steps                  //warmup步数
  --mixed_precision                  //混合精度
  --max_train_steps                  //最大训练轮次
  --validation_prompt                //验证的prompt
  --validation_epochs                //每epochs验证一次
  --validation_steps                 //每steps验证一次(仅controlnet微调脚本使用)
  --enable_npu_flash_attention       //开启NPU的FlashAttention
  --seed                             //随机数种子
  --output_dir                       //模型输出的路径
  --gradient_accumulation_steps      //梯度累计步数
  --validation_image                 //验证使用的图片(仅controlnet微调脚本使用)
 ```

#### 训练结果


##### 性能

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| GPU | 8p |    LoRA    | 23.38 |     7      | fp16 | 2.1 | ✔ |
| Atlas A2 |8p |    LoRA    | 28.75 |     7      | fp16 | 2.1 | ✔ |
| GPU | 8p | Controlnet | 32.5  |     5      | fp16 | 2.1 | ✔ |
| Atlas A2 |8p | Controlnet | 28.42 |     5      | fp16 | 2.1 | ✔ |
| GPU | 8p |  Finetune  | 142.7 |     24     | fp16 | 2.1 | ✔ |
| Atlas A2 |8p |  Finetune  | 172.9 |     24     | fp16 | 2.1 | ✔ |

### 推理任务
本任务主要以预训练模型为主，展示推理任务，包括单卡预训练推理。
#### 开始推理
1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称} 
   ```

2. 运行推理的脚本。

- 单机单卡推理
- 推理前加载环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

- 调用推理脚本
  ```shell
  python examples/SDXL/sdxl_text2img_lora_infer.py        # 混精fp16 文生图lora微调任务推理
  python examples/SDXL/sdxl_text2img_controlnet_infer.py  # 混精fp16 文生图controlnet微调任务推理
  python examples/SDXL/sdxl_text2img_infer.py             # 混精fp16 文生图全参微调任务推理
  python examples/SDXL/sdxl_img2img_infer.py              # 混精fp16 图生图微调任务推理
  ```

# SDXL_Turbo

## 准备训练环境

### 安装模型环境

  **表 3**  三方库版本支持表

  | 三方库    | 支持版本  |
  | :--------: | :-------------: |
  | PyTorch | 2.1.0 |
  | diffusers | 0.28.0 |
  | accelerate | 0.29.3 | 
  | deepspeed | 0.14.1|

  
### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表4中软件版本。

  **表 4**  昇腾软件版本支持表

  |     软件类型          |   支持版本   |
  |:-----------------:|:--------:|
  | FrameworkPTAdaper |   在研版本   |
  |       CANN        | 在研版本  |
  |      昇腾NPU固件      | 在研版本 |
  |      昇腾NPU驱动      | 在研版本 |

### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   stabilityai/sdxl-turbo #预训练模型
   ```
   
## 快速开始
### 推理任务

本任务主要提供**混精fp16**的**单卡**推理脚本。

#### 开始推理
- 推理前加载环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
- 调用推理脚本
  ```shell
  python examples/SDXL_Turbo/sdxl_turbo_infer.py   #单卡推理，混精fp16
  ```

# AnimateDiff

## 准备训练环境

### 安装模型环境
  **表 5**  三方库版本支持表

  | 三方库    |  支持版本  |
  | :--------: |:------:|
  | PyTorch | 2.1.0  |
  | diffusers | 0.27.2 |
  | accelerate | 0.29.3 | 
  | deepspeed | 0.14.1 |

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表4中软件版本。

  **表 6**  昇腾软件版本支持表

  |     软件类型          |   支持版本   |
  |:-----------------:|:--------:|
  | FrameworkPTAdaper |   在研版本   |
  |       CANN        | 在研版本  |
  |      昇腾NPU固件      | 在研版本 |
  |      昇腾NPU驱动      | 在研版本 |

### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   guoyww/animatediff-motion-adapter-v1-5-2
   SG161222/Realistic_Vision_V5.1_noVAE
   wangfuyun/AnimateLCM
   emilianJR/epiCRealism
   guoyww/animatediff-motion-lora-zoom-out
   guoyww/animatediff-motion-lora-pan-left
   ```
### 获取推理需要的数据

在执行AnimateDiffvideotovideopipeline.py脚本时需要手动下载gif作为初始的图片传入其中，可以参考这个文档 https://huggingface.co/docs/diffusers/api/pipelines/animatediff 获取下面的gif资源。
```shell

animatediff-vid2vid-input-1.gif

```

## 快速开始
### 推理任务

本任务主要提供**混精fp16**的**单卡**推理脚本。

#### 开始推理
- 将模型上述路径传入到examples/AnimateDiff目录下的py脚本里面:
AnimateDiffpipeline.py
AnimateDiffvideotovideopipeline.py
AnimateLCM.py
FreeInit.py
motion_lora.py
motion_lora_with_peft.py`

- 算子问题规避:
  - 执行 pip show diffusers 找到diffusers包的路径path
  - 执行 cd path/diffusers/pipelines/ 找到下面的free_init_utils.py
  - 在free_init_utils.py文件的118行，找到_apply_freq_filter函数，修改为下面的代码。
    <details>
    <summary>点击展开代码</summary>
    
    ```python
    def _apply_freq_filter(self, x: torch.Tensor, noise: torch.Tensor, low_pass_filter: torch.Tensor) -> torch.Tensor:
        r"""Noise reinitialization."""
    
        x=x.to("cpu")
        noise=noise.to("cpu")
        low_pass_filter=low_pass_filter.to("cpu")

        # FFT
        x_freq = fft.fftn(x, dim=(-3, -2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
        noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
        noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

        # frequency mix
        high_pass_filter = 1 - low_pass_filter
        x_freq_low = x_freq * low_pass_filter
        noise_freq_high = noise_freq * high_pass_filter
        x_freq_mixed = x_freq_low + noise_freq_high  # mix in freq domain

        # IFFT
        x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
        x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real
    
        x_mixed = x_mixed.to("npu")
        return x_mixed
    ```  
    </details>

- 调用推理脚本
  ```shell
  bash examples/AnimateDiff/AnimateDiff_infer.sh  #单卡推理，混精fp16
  ```

# DiT

## 准备环境

### 安装模型环境

**表 7**  三方库版本支持表

|  三方库   | 支持版本 |
| :-------: | :------: |
|  PyTorch  |  2.1.0   |
| diffusers |  0.28.0  |

### 安装昇腾环境

 请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表8中软件版本。

  **表 8**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 在研版本 |
|       CANN        | 在研版本 |
|    昇腾NPU固件    | 在研版本 |
|    昇腾NPU驱动    | 在研版本 |

## 快速开始

### 推理任务

本任务主要提供**混精fp16**的**单卡**推理脚本。

#### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   facebook/DiT-XL-2-256
   facebook/DiT-XL-2-512
   ```

#### 开始推理

- 推理前加载环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
- 调用推理脚本
  ```shell
  python examples/DiT/dit_infer.py   #单卡推理，混精fp16
  ```




# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.05.20：SDXL、SDXL_turbo、Animatediff流水线样例首次发布。

# FAQ


