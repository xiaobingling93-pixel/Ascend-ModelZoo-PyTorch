# OpenSoraPlan1.1 for PyTorch

**注意**： 本仓库OpenSoraPlan1.1模型将不再进行维护，请使用[MindSpeed-MM](https://gitcode.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.3)

# 目录
<!-- TOC -->
* [OpenSoraPlan1.1 for PyTorch](#opensoraplan11-for-pytorch)
* [目录](#目录)
* [简介](#简介)
  * [模型介绍](#模型介绍)
  * [支持任务列表](#支持任务列表)
  * [代码实现](#代码实现)
* [准备训练环境](#准备训练环境)
  * [安装模型环境](#安装模型环境)
  * [安装昇腾环境](#安装昇腾环境)
* [LatteT2V](#lattet2v)
  * [训练数据集准备](#训练数据集准备)
  * [准备预训练模型](#准备预训练模型)
  * [快速开始](#快速开始)
    * [训练任务](#训练任务)
      * [单机训练](#单机训练)
      * [多机训练](#多机训练)
      * [训练结果](#训练结果)
        * [性能](#性能)
    * [在线推理任务](#在线推理任务)
      * [开始推理](#开始推理)
* [公网地址说明](#公网地址说明)
* [变更说明](#变更说明)
  * [变更](#变更)
* [FAQ](#faq)
<!-- TOC -->

# 简介

## 模型介绍

Open-Sora-Plan是由北大技术团队推出的项目，旨在通过开源框架复现 OpenAI
Sora。作为基础开源框架，它支持视频生成模型的训练，包括无条件视频生成、类别视频生成和文本到视频生成。
本仓库主要将Open-Sora-Plan多个任务迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型。

|    模型    | 任务列表 | 是否支持 |
|:--------:|:----:|:----:|
| LatteT2V | 单机训练 |  ✔   |
| LatteT2V | 多机训练 |  ✔   |
| LatteT2V | 在线推理 |  ✔   |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/PKU-YuanGroup/Open-Sora-Plan
  commit_id=2a8b2328a5fcc0108fb5444b010f7e1ae0b4cb7b
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/
  ```

# 准备训练环境

## 安装模型环境

**表 1**  三方库版本支持表

|     三方库      |  支持版本(PT2.1)  |  支持版本(PT2.4)  |
|:------------:|:------:|:------:|
|   PyTorch    | 2.1.0  | 2.4.0  |
|  diffusers   | 0.27.2 | 0.27.2 |
|  accelerate  | 0.28.0 | 0.29.3 |
|  deepspeed   | 0.12.6 | 0.15.3 |
| transformers | 4.39.1 | 4.40.1 |
|    decord    | 0.6.0  | 0.6.0  |


1. 安装decord。
   - X86等架构环境可以在步骤3自动安装。
   - 对arm等架构，如无法自动获取，需要参考[源仓README](https://github.com/dmlc/decord/blob/master/README.md#install-from-source)从源码安装。

2. 安装mindspeed。
   - 联网条件下，可以直接克隆源仓。
     ```shell
     git clone https://gitcode.com/ascend/MindSpeed.git
     pip install -e MindSpeed
     ```
   - 离线条件下，可以[下载源仓代码](https://gitcode.com/ascend/MindSpeed/releases)，再执行安装。
     ```shell
     pip install -e MindSpeed-v1.1.0
     ```

3. 在模型根目录下执行以下命令，安装其他依赖。

    ```shell
    pip install -e .   # 安装本地OpenSoraPlan代码仓
    # 若使用PyTorch 2.4请另外使用requirements_2_4.txt
    pip install -r requirements_2_4.txt
    ```

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)
》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|        软件类型        | 支持版本 |
  |:------------------:|:----:|
| FrameworkPTAdapter | 在研版本 |
|        CANN        | 在研版本 |
|      昇腾NPU固件       | 在研版本 | 
|      昇腾NPU驱动       | 在研版本 |

# LatteT2V

## 训练数据集准备

用户需自行获取并解压mixkit2数据集，以及对应帧数的标注json，放置到`OpenSoraPlan1.1/dataset`目录下。
数据和标注可以从huggingface的LanguageBind/Open-Sora-Plan-v1.1.0数据集文件的all_mixkit和anno_jsons中获取。

数据结构如下：

   ```
   OpenSoraPlan1.1
   ├── dataset
      ├── mixkit2
          ├── Airplane
          ├── Baby
          ├── ...
          └── video_mixkit_65f_54735.json
   ```

## 准备预训练模型

1. 下载主要预训练模型。
   - 联网情况下，预训练模型会自动下载。

   - 无网络时，用户可访问huggingface官网自行下载主要权重，文件namespace如下：

      ```
      DeepFloyd/t5-v1_1-xxl               # t5模型
      LanguageBind/Open-Sora-Plan-v1.1.0  # 预训练权重(含3D VAE模型和LatteT2V模型)
      ```

2. 将下载好的预训练模型放在本工程目录下，组织结构如下：
   ```
   OpenSoraPlan1.1
   ├── DeepFloyd
   │   ├── t5-v1_1-xxl
   │   │   ├── config.json
   │   │   ├── pytorch_model-00001-of-00002.bin
   │   │   ├── ...
   │   LanguageBind
   │   ├── Open-Sora-Plan-v1.1.0
   │   │   ├── 221x512x512
   │   │   ├── 65x512x512
   │   │   └── vae
   ```

3. 模型还用到了torchvision提供的默认VGG16预训练权重。
   - 联网情况下，预训练模型会自动下载。
   - 无网络时，用户可自行下载权重，并放置到默认的`~/.cache/torch/hub/checkpoints`目录下。下载信息见`torchvision.models.vgg.VGG16_Weights.IMAGENET1K_V1`。

## 快速开始

### 训练任务

本任务主要提供**混精bf16 单机8卡**、**混精bf16 双机32卡**两种训练脚本，用于65帧分辨率为512x512的文生视频训练。

#### 单机训练

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行预训练脚本。

   该模型支持单机8卡训练，执行以下命令执行训练。

    ```shell
    bash scripts/text_condition/train_videoae_65x512x512_16.sh # 8卡训练，混精bf16
    ```

   模型训练脚本参数说明如下。
   ```shell
       --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \ // deepspeed配置文件
       opensora/train/train_t2v.py \                                          // 训练启动脚本
       --model LatteT2V-XL/122 \                                              // 训练模型
       --text_encoder_name DeepFloyd/t5-v1_1-xxl \                            // 文本编码器
       --cache_dir "./cache_dir" \                                            // 下载缓存目录
       --dataset t2v \                                                        // 数据集类型
       --ae CausalVAEModel_4x8x8 \                                            // 图片/视频预训练模型
       --ae_path "LanguageBind/Open-Sora-Plan-v1.1.0/vae" \                   // vae预训练文件路径
       --video_data "scripts/train_data/video_data.txt" \                     // 视频数据路径文件
       --use_img_from_vid \                                                   // 训练图片来自视频
       --sample_rate 1 \                                                      // 采样率
       --num_frames 65 \                                                      // 训练帧数
       --max_image_size 512 \                                                 // 图像/视频最大尺寸
       --gradient_checkpointing \                                             // 是否重计算
       --attention_mode math \                                                // attention的类型
       --train_batch_size=2 \                                                 // 训练的批大小
       --dataloader_num_workers 4 \                                           // 数据处理线程数
       --gradient_accumulation_steps=1 \                                      // 梯度累计步数
       --max_train_steps=1000000 \                                            // 最大训练步数
       --learning_rate=2e-05 \                                                // 学习率
       --lr_scheduler="constant" \                                            // 学习率调度策略
       --lr_warmup_steps=0 \                                                  // 学习率预热步数
       --mixed_precision="bf16" \                                             // 混精训练的数据类型
       --report_to="tensorboard" \                                            // 记录方式
       --checkpointing_steps=500 \                                            // 检查点步数
       --output_dir="65x512x512_10node_bs2_lr2e-5_16img" \                    // 输出的路径
       --allow_tf32 \                                                         // 使用tf32训练
       --use_deepspeed \                                                      // 使用deepspeed训练
       --model_max_length 300 \                                               // 文本最大长度
       --use_image_num 16 \                                                   // 训练使用图片的数量
       --enable_tiling \                                                      // 启用平铺
       --pretrained LanguageBind/Open-Sora-Plan-v1.1.0/65x512x512/diffusion_pytorch_model.safetensors // 预训练模型
   ```

#### 多机训练

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 修改配置文件。
    
    修改`scripts/accelerate_configs/multi_node.yaml`，配置说明如下：
    ```yaml
    compute_environment: LOCAL_MACHINE                            # 计算环境
    distributed_type: DEEPSPEED                                   # 分布式类型
    deepspeed_config:                                             # deepspeed配置
     deepspeed_multinode_launcher: standard                       # deepspeed多节点启动器
     deepspeed_config_file: scripts/accelerate_configs/zero2.json # deepspeed配置文件
    main_process_ip: 10.0.10.100                                  # 主进程IP
    main_process_port: 29502                                      # 主进程端口
    main_training_function: main                                  # 主训练函数
    num_machines: 2                                               # 机器数量
    num_processes: 32                                             # 进程数量
    ```
3. 运行预训练脚本。

    分别在双机集群中执行以下脚本：

    ```shell
    bash scripts/text_condition/multi_node.sh # 32卡训练，混精bf16
    ```
    相对单机训练脚本，多机额外参数说明如下：

    ```shell
    --machine_rank 0 \ # 机器编号
    ```

#### 训练结果

##### 性能

|         芯片         | 卡数 | 单步迭代时间(s/step) | batch_size | AMP_Type | Torch_Version |
|:------------------:|:--:|:--------------:|:----------:|:--------:|:-------------:|
|        竞品A         | 8p |      9.19      |     2      |   bf16   |      2.1      |
| Atlas 900 A2 Box16 | 8p |      9.40      |     2      |   bf16   |      2.1      |
### 在线推理任务

#### 开始推理

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行推理脚本。

   该模型支持单卡文生视频在线推理。

   - 执行单卡推理。
   
    ```shell
    bash scripts/text_condition/sample_video_65x512x512.sh
    ```

   - 模型在线推理python脚本参数说明如下。

   ```shell
    python opensora/sample/sample_t2v.py \               # 启动推理脚本
    --model_path LanguageBind/Open-Sora-Plan-v1.1.0 \    # LatteT2V预训练权重父路径
    --version 65x512x512 \                               # LatteT2V预训练权重子路径
    --num_frames 65 \                                    # 视频帧数
    --height 512 \                                       # 视频高度像素数
    --width 512 \                                        # 视频宽度像素数
    --cache_dir "./cache_dir" \                          # 缓存路径
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \          # 文本编码模型命名空间
    --text_prompt examples/prompt_list_0.txt \           # 文本提示文件路径
    --ae CausalVAEModel_4x8x8 \                          # 视频压缩模型
    --ae_path "LanguageBind/Open-Sora-Plan-v1.1.0/vae" \ # 视频压缩模型预训练权重路径
    --save_img_path "./sample_video_65x512x512" \        # 生成的视频文件路径
    --fps 24 \                                           # 生成的视频帧率
    --guidance_scale 7.5 \                               # 指导尺度
    --num_sampling_steps 150 \                           # 采样步数
    --enable_tiling                                      # 启用平铺
   ```

# 公网地址说明

代码涉及公网地址参考 [public_address_statement.md](public_address_statement.md)。

# 变更说明

## 变更

2024.06.20: LatteT2V bf16训练任务首次发布。

2024.07.24: LatteT2V fp16推理任务、多机训练任务首次发布。

# FAQ

暂无。

