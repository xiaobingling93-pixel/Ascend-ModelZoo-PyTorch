# Diffusers0.25.0 for Pytorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [SDXL](#SDXL)   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [预训练任务](#预训练任务)
          - [微调任务](#微调任务)
          - [推理任务](#推理任务)
-   [SVD（在研版本）](#SVD（在研版本）)   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [推理任务](#推理任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

扩散模型 (Diffusion Models) 是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是Huggingface发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频甚至分子的3D结构。套件包含基于扩散模型的多种个模型，提供了各种下游任务的训练与推理的实现。
本仓库主要将SDXL、SVD模型的多个任务迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|  模型  |    任务列表    | 是否支持 |
|:----:|:----------:|:-----:|
| SDXL |    预训练     | ✔ |
| SDXL |    Lora    | ✔ |
| SDXL | Controlnet | ✔ |
| SDXL |   文生图推理    | ✔ |
| SVD  |   文生视频推理   | ✔ |


## 代码实现

- 参考实现：

  ```
  url=https://github.com/huggingface/diffusers
  commit_id=7f551e29ff4ad05615cb38530a8940811f9e5936
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/diffusion/
  ```


# SDXL

## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  | 三方库    | 支持版本  |
  | :--------: | :-------------: |
  | PyTorch | 2.1.0 |
  | diffusers | 0.25.0 |
  | accelerate | 0.25.0 | 
  | deepspeed | 0.12.6|


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```python
   pip install -e .                                                # 安装本地diffusers代码仓
   pip install -r examples/text_to_image/requirements_sdxl.txt     # 安装对应依赖
   ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

  | 软件类型   | 支持版本  |
  | :--------: | :-------------: |
  | FrameworkPTAdapter | 6.0.RC1 |
  | CANN | 8.0.RC1 |
  | 昇腾NPU固件 | 24.1.RC1 | 
  | 昇腾NPU驱动 | 24.1.RC1 |

  

### 准备数据集

#### 预训练数据集准备

1. 用户需自行获取并解压LAION_5B数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径。

   ```shell
   test/train_8p_text_to_image_sdxl_pretrain_fp16.sh
   test/train_8p_text_to_image_sdxl_pretrain_bf16.sh
   ```

   数据结构如下：

   ```
   $LAION5B
   ├── 000000000.jpg
   ├── 000000000.json
   ├── 000000000.txt
   └── ...
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。
   > 用户可获取COCO数据集替换Laion_5B数据集，需按要求预处理为以上格式，例如以下处理脚本：
   > ```python
   > import torchvision.datasets as dset
   > dt = dset.CocoCaption(root="coco_path",
   >                       annFile="coco_path/annotations/captions_train2017.json")
   > path = "data_path"
   > for index, target in enumerate(dt):
   >     target[0].save(path + str(index) + ".jpg")
   >     with open(path + str(index) + ".txt", mode="w+", encoding="utf-8") as f:
   >         f.writelines(target[1][0])
   >     f.close() 
   > ```

#### 微调数据集准备
##### LoRA微调

   1. 联网情况下，数据集会自动下载。
   2. 无网络情况下，用户需自行获取pokemon-blip-captions数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径。

   ```shell
   test/train_8p_sdxl_lora.sh
   test/train_8p_sdxl_lora_deepspeed.sh
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
   test/train_8p_controlnet_sdxl.sh 
   test/train_8p_controlnet_sdxl_deepspeed.sh
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

   

### 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   stabilityai/stable-diffusion-xl-base-1.0 #预训练模型
   madebyollin/sdxl-vae-fp16-fix #vae模型
   ```

3. 获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径。
   ```shell
   test/train_8p_text_to_image_sdxl_pretrain_fp16.sh
   test/train_8p_text_to_image_sdxl_pretrain_bf16.sh
   test/train_8p_sdxl_lora.sh
   test/train_8p_sdxl_lora_deepspeed.sh
   test/train_8p_controlnet_sdxl.sh 
   test/train_8p_controlnet_sdxl_deepspeed.sh
   ```
## 快速开始

### 预训练任务

本任务主要提供**混精fp16**和**混精bf16**两种**8卡**训练脚本，默认使用**deepspeed**分布式训练。

#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行预训练脚本。

   该模型支持单机8卡训练。
   
  
   - 单机8卡训练
   
     ```shell
     bash test/train_8p_text_to_image_sdxl_pretrain_fp16.sh # 8卡训练，混精fp16
     bash test/train_8p_text_to_image_sdxl_pretrain_fp16.sh --max_train_steps=200 # 8卡性能，混精fp16
     
     bash test/train_8p_text_to_image_sdxl_pretrain_bf16.sh # 8卡训练，混精bf16
     bash test/train_8p_text_to_image_sdxl_pretrain_bf16.sh --max_train_steps=200 # 8卡性能，混精bf16
     ```
      > 注：
      > 1. 预训练默认开启动态分辨率，如需使用静态分辨率需要使用:
      >     1. 数据集原图像分辨率大小足够大；
      >     2. 设置最大、最小动态分辨率桶均为1024。
      > 2. 根据以下指引修改环境中deepspeed仓库的参数以获得性能加速：
      > ```python
      >    # 位于deepspeed/runtime/bf16_optimizer.py文件中第66行:
      >    self.nccl_start_alignment_factor = 2 #修改为：
      >    self.nccl_start_alignment_factor = 128 
      >    # 位于deepspeed/runtime/zero/stage_1_and_2.py文件中第276行：
      >    self.nccl_start_alignment_factor = 2 #修改为：
      >    self.nccl_start_alignment_factor = 128 
      > ```
      > 3. 训练完成后，模型权重文件，训练精度和性能等信息保存在`test/output`路径下。
      
   - 模型训练python训练脚本参数说明如下。
   
   ```shell
   examples/text_to_image/train_text_to_image_sdxl_pretrain.py
   --max_train_steps                   //训练步数
   --pretrained_model_name_or_path     //预训练模型名称或者地址
   --dataset_name                      //加载数据集的方式，从官网或者本地cache中读取数据
   --vae_name                          //预训练vae模型名称或者地址
   --dataset_config_name               //数据集配置     
   --train_batch_size                  //设置batch_size
   --image_column                      //图片所在列
   --caption_column                    //图片caption所在列
   --max_train_samples                 //最大训练样本数
   --validation_prompts                //验证提示词
   --output_dir                        //输出路径
   --resolution                        //分辨率
   --num_train_epochs                  //训练epoch数
   --gradient_accumulation_steps       //梯度累计步数
   --mixed_precision                   //精度模式
   --num_train_epochs                  //训练回合数
   --enable_bucket                     //启动动态分辨率
   --max_bucket_reso                   //设置最大动态分辨率桶，默认2048
   --min_bucket_reso                   //设置最小动态分辨率桶，默认512
   ```
   
#### 训练结果


##### 性能
| 芯片 | 卡数 | FPS | batch_size | AMP_Type | Torch_Version |
|:---:|:---:|:---:|:---:|:---:|:---:|
| GPU | 8p | 20.65 | 4 | bf16 | 2.1 |
| Atlas A2 | 8p | 17.19 | 4 | bf16 | 2.1 |
| GPU | 8p | 19.94 | 4 | fp16 | 2.1 |
| Atlas A2 | 8p | 17.23 | 4 | fp16 | 2.1 |

### 微调任务
本任务主要提供LoRA和Controlnet两种微调下游任务的8卡训练脚本，包括使用和不使用deepspeed分布式训练。

#### 开始训练
   

1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称} 
   ```


2. 运行训练的脚本。
- 单机八卡微调
  ```shell
  bash test/train_8p_controlnet_sdxl_deepspeed.sh      #8卡deepspeed训练 sdxl_controlnet fp16
  bash test/train_8p_sdxl_lora_deepspeed.sh            #8卡deepspeed训练 sdxl_lora fp16
  bash test/train_8p_controlnet_sdxl.sh                #8卡训练 sdxl_controlnet fp16
  bash test/train_8p_sdxl_lora.sh                      #8卡训练 sdxl_lora fp16
  ```
 - 微调脚本参数说明如下
 ```shell
  examples/text_to_image/train_text_to_image_lora_sdxl.py or examples/text_to_image/train_controlnet_sdxl.py
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
  --seed                             //随机数种子
  --output_dir                       //模型输出的路径
  --gradient_accumulation_steps      //梯度累计步数
  --validation_image                 //验证使用的图片(仅controlnet微调脚本使用)
 ```

#### 训练结果


##### 性能
| 芯片 | 卡数 | 任务 | FPS | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPU | 8p | LoRA | 23.38 | 7 | fp16 | 2.1 | ✔ |
| Atlas A2 |8p | LoRA  | 28.75 | 7 | fp16 | 2.1 | ✔ |
| GPU | 8p | LoRA | 19.48 | 3 | fp16 | 2.1 | ✘ |
| Atlas A2 |8p | LoRA  | 18.76 | 3 | fp16 | 2.1 | ✘ |
| GPU | 8p | Controlnet | 32.5 | 5 | fp16 | 2.1 | ✔ |
| Atlas A2 |8p | Controlnet  | 28.42 | 5 | fp16 | 2.1 | ✔ |
| GPU | 8p | Controlnet | 20.52 | 2 | fp16 | 2.1 | ✘ |
| Atlas A2 |8p | Controlnet | 32.69 | 2 | fp16 | 2.1 | ✘ |

### 推理任务
本任务主要以预训练模型为主，展示推理任务，包括单卡预训练推理。
#### 开始推理
1. 进入解压后的源码包根目录。

      ```
   cd /${模型文件夹名称} 
   ```


2. 运行推理的脚本。

- 单机单卡推理
  ```shell
  bash test/infer_full_1p_text_to_image_sdxl_fp16.sh # 混精fp16 预训练任务推理
  ```
- 微调脚本参数说明如下
   ```shell
   examples/text_to_image/infer_text_to_image.py
   --mixed_precision          //混合精度
   --ckpt_path                //模型地址
   --output_path              //输出地址
   --device_id                //设备id
   ```


# SVD（在研版本）


## 准备训练环境

### 安装模型环境

  **表 3**  三方库版本支持表

  |     三方库     |  支持版本  |
  |:-----------:|:------:|
  |   PyTorch   | 2.1.0  |
  | TorchVision | 0.16.0 |
  |  diffusers  | 0.25.0 |
  | accelerate  | 0.27.2 |

  在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  pip install -e .                              # 安装本地diffusers代码仓
  cd examples/stable_video_diffusion/           # 根据下游任务安装对应依赖
  pip install -r requirements_svd.txt 
  ```
  
### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表4中软件版本。

  **表 4**  昇腾软件版本支持表

  |     软件类型          |   支持版本   |
  |:-----------------:|:--------:|
  | FrameworkPTAdaper |   在研版本   |
  |       CANN        | 在研版本  |
  |      昇腾NPU固件      | 在研版本 |
  |      昇腾NPU驱动      | 在研版本 |


### 准备数据集
  1. 用户自行获取i2vgen-xl数据集，放到模型目录下，并重命名为`svd_testdata`

  2. 用户自行准备数据路径txt文件，存放测试图片的文件名

     参考数据集结构为

     ```
     diffusers0.25.0
     ├── svd_testdata
     |   ├── img_0001.jpg
     |   ├── img_0002.jpg
     |   ├── ...
     |   ├── imglist.txt
     ```
     > **说明：** 
     该数据集的推理过程脚本只作为一种参考示例。
   

### 获取预训练权重

   1. 联网情况下，预训练模型会自动下载。

   2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

      ```
      stabilityai/stable-video-diffusion-img2vid-xt
      ```

   3. 获取对应的预训练模型后，在shell启动脚本中将`ckpt_path`参数，设置为本地预训练模型路径，填写一级目录。


## 快速开始
### 推理任务

本任务主要提供**混精fp16**的**单卡**和**8卡**推理脚本。

#### 开始训练
   1. 进入源码根目录。

      ```
      cd /${模型文件夹名称}
      ```

   2. 运行推理脚本。

      该模型支持单机单卡和8卡推理。
      - 单机单卡推理

        ```shell
        bash test/infer_full_1p_svd_fp16.sh --ckpt_path=xxx --test_data_dir=xxx --test_file=xxx # 单卡推理，混精fp16
        ```

      - 单机8卡推理

        ```shell
        bash test/infer_full_8p_svd_fp16.sh --ckpt_path=xxx --test_data_dir=xxx --test_file=xxx # 八卡推理，混精fp16
        ```

   模型推理脚本参数说明如下。
   
   ```
   infer_full_8p_svd_fp16.sh
   --ckpt_path                             // 模型权重加载地址
   --batch_size                            // 推理的图像全局批大小
   --test_file                             // 测试图片路径文件
   --test_data_dir                         // 测试数据集存放目录
   
   test_stable_video_diffusion.py
   --ckpt                                  // 模型权重加载地址
   --global-batch-size                     // 推理的图像全局批大小
   --test-file                             // 测试图片路径文件
   --test-data-dir                         // 测试数据集存放目录
   --export-video                          // 是否导出视频文件
   --num-frames                            // 生成帧数
   --seed                                  // 随机种子
   --image-size                            // 输入图片resize分辨率
   --num-workers                           // 读取数据集的线程数
   --output-dir                            // 生成文件的保存目录
   --eval-metrics                          // 是否评估推理精度
   --benchmark-dir                         // 评估对比图片的目录
   ```

#### 推理结果
##### 性能

| 芯片       | 卡数 | Denoise FPS | batch_size |  AMP_Type  | Torch_Version | 
|----------|:--:|:-----------:|:----------:|:----------:|:-------------:|
| GPU      | 8p |    4.88     |     8      |    fp16    |      2.1      |
| Atlas A2 | 8p |    4.06     |     8      |    fp16    |      2.1      |

 > 注：denoise FPS是根据denoise的单步时间计算的FPS（帧数/时间），等于BatchSize/denoise step time.

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.01.30：SDXL fp16预训练任务首次发布。

2024.01.31：SDXL微调任务首次发布。

2024.02.05：SDXL bf16预训练任务首次发布。

2024.03.06：SDXL 预训练任务添加deepspeed config。

2024.03.14：SVD 推理任务首次发布。

# FAQ



