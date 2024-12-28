# OpenSoraPlan1.0 for PyTorch

**注意**： 本仓库OpenSoraPlan1.0模型将不再进行维护，请使用[MindSpeed-MM](https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.3)

# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)

-   [准备训练环境](#准备训练环境)
-   [VideoGPT](#VideoGPT)   
    -  [训练数据集准备](#训练数据集准备)
    -  [快速开始](#快速开始)
          - [训练任务](#训练任务)
          - [性能展示](#性能展示)
-   [LatteT2V](#LatteT2V)
    -  [训练数据集准备](#训练数据集准备)
    -  [准备预训练模型](#准备预训练模型)
    -  [快速开始](#快速开始)
          - [训练任务](#训练任务)
          - [性能展示](#性能展示)
          - [在线推理任务](#在线推理任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

Open-Sora-Plan是由北大技术团队推出的项目，旨在通过开源框架复现 OpenAI Sora。作为基础开源框架，它支持视频生成模型的训练，包括无条件视频生成、类别视频生成和文本到视频生成。
本仓库主要将Open-Sora-Plan多个任务迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|  模型  |    任务列表    | 是否支持 |
|:----:|:----------:|:-----:|
| VideoGPT |    训练     | ✔ |
| LatteT2V |    训练     | ✔ |
| LatteT2V |  在线推理   | ✔ |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/PKU-YuanGroup/Open-Sora-Plan
  commit_id=a7375034586fea20b4aa14bc17c58adbaeeef32f
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/
  ```

# 准备训练环境

## 安装模型环境

  **表 1**  三方库版本支持表

  | 三方库    | 支持版本(PT2.1)  | 支持版本(PT2.4)  |
  |:------------:|:------:|:------:|
  | PyTorch      | 2.1.0  | 2.4.0  |
  | diffusers    | 0.27.2 | 0.27.2 |
  | accelerate   | 0.28.0 | 0.29.3 |
  | deepspeed    | 0.12.6 | 0.15.3 |
  | transformers | 4.39.1 | 4.40.1 |
  | decord       | 0.6.0  | 0.6.0  |


  在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

   ```shell
   pip install -e .   # 安装本地OpenSoraPlan代码仓
   # 若使用PyTorch 2.4请另外使用requirements_2_4.txt
   pip install -r requirements_2_4.txt
   ```

  注: 模型依赖decord需编译安装，根据原仓安装https://github.com/dmlc/decord

## 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                

  **表 2**  昇腾软件版本支持表

  | 软件类型   |   支持版本   |
  | :--------: |:--------:|
  | FrameworkPTAdapter |   在研版本   |
  | CANN |   在研版本   |
  | 昇腾NPU固件 |   在研版本   | 
  | 昇腾NPU驱动 | 在研版本 |

## 训练数据集准备

   用户需自行获取并解压MSRVTT数据集，放置到`OpenSoraPlan1.0/dataset`目录下。

   数据结构如下：

   ```
   OpenSoraPlan1.0
   ├── dataset
      ├── MSRVTT
          ├── annotation
          ├── high-quality
          ├── structured-symlinks
          └── video
   ```

# VideoGPT

## 训练数据集准备

   用户需在以下启动shell脚本中将`data_path`参数设置为本地数据集的绝对路径。

   ```shell
   bash scripts/videogpt/train_videogpt.sh
   ```

## 快速开始

### 训练任务

  本任务主要提供**混精bf16**一种**8卡**训练脚本，默认使用**DDP**分布式训练。

#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行预训练脚本。

   该模型支持单机8卡训练。
   
  
   - 单机8卡训练
   
    ```shell
    bash scripts/videogpt/train_videogpt.sh # 8卡训练，混精bf16
    ```
   
   - 模型训练python训练脚本参数说明如下。
   
   ```shell
   bash scripts/videogpt/train_videogpt.sh
   --max_steps                         //训练步数
   --data_path                         //加载数据集地址
   --per_device_train_batch_size       //设置batch_size
   --save_strategy                     //保存策略
   --learning_rate                     //学习率大小
   --lr_scheduler_type                 //学习率策略
   --max_train_samples                 //最大训练样本数
   --output_dir                        //输出路径
   --resolution                        //分辨率
   --gradient_accumulation_steps       //梯度累计步数
   --save_total_limit                  //保存限制次数
   --logging_steps                     //结果打印次数
   --downsample                        //下采样率
   --n_res_layers                      //残差层数
   --embedding_dim                     //嵌入层维度
   --n_hiddens                         //注意力头数
   --n_codes                           //codebook维度
   --sequence_length                   //帧数
   --report_to                         //记录方式
   --bf16                              //bf16精度模式
   --fp16                              //fp16精度模式
   --dataloader_num_workers            //设置dataloader workers数量  
   
   ```
   注：当前模型不支持断点续训，因此暂无相关参数
   
#### 性能展示

##### 性能

| 芯片 | 卡数 | 单步迭代时间(s/step)  | batch_size | 帧数 | AMP_Type | Torch_Version |
|:---:|:---:|:----:|:----------:|:---:|:---:|:---:|
| 竞品A | 8p | 0.62 |     1      | 240 | bf16 | 2.1 |
| Atlas 800T A2 | 8p | 1.00 |     1      | 240 | bf16 | 2.1 |

# LatteT2V

## 训练数据集准备

在源码根目录下进行数据集预处理。

```shell
cd OpenSoraPlan1.0/
python dataset/collate_msrvtt_dataset.py -d dataset/MSRVTT -o dataset/msrvtt
python dataset/preprocess_msrvtt.py --data_path dataset/msrvtt/train/annotations.json   # 生成最终的标注csv文件
```

-d: MSRVTT原数据集的路径；
-o: MSRVTT处理后数据集的路径；
--data_path: MSRVTT处理后数据集的路径。


处理后数据结构如下：

    ```
    msrvtt
    ├── train
    │   ├── videos
    │   │   ├── video0.mp4
    │   │   ├── video1.mp4
    │   │   └── ...
    │   └── annotation.json
    ├── val
    ├── test
    └── ...
    
    ```

## 准备预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   DeepFloyd/t5-v1_1-xxl               # t5模型
   LanguageBind/Open-Sora-Plan-v1.0.0  # 预训练权重(含3D VAE模型和LatteT2V模型)
   ```

3. 将下载好的预训练模型放在本工程目录下，组织结构如下：
   ```
   $OpenSoraPlan1.0
   ├── DeepFloyd
   │   ├── t5-v1_1-xxl
   │   │   ├── config.json
   │   │   ├── pytorch_model-00001-of-00002.bin
   │   │   ├── ...
   │   LanguageBind
   │   ├── Open-Sora-Plan-v1.0.0
   │   │   ├── 17x256x256
   │   │   ├── 65x256x256
   │   │   ├── 65x512x512
   │   │   └── vae
   ```

## 快速开始

### 训练任务

本任务主要提供**混精bf16**一种**8卡**训练脚本，17帧分辨率为256x256的文生视频训练。

#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行预训练脚本。

   该模型支持单机8卡训练。
  
   - 单机8卡训练
   
    ```shell
    bash scripts/text_condition/train_videoae_17x256x256.sh # 8卡训练，混精bf16
    ```
   
   - 模型训练python训练脚本参数说明如下。
    ```shell
      --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \          //deepspeed配置文件
      opensora/train/train_t2v.py \                                                   //训练启动脚本
      --model LatteT2V-XL/122 \                                                       //训练模型
      --text_encoder_name DeepFloyd/t5-v1_1-xxl \                                     //文本编码器
      --dataset t2v \                                                                 //数据集类型
      --ae CausalVAEModel_4x8x8 \                                                     //视频/图片压缩模型
      --ae_path LanguageBind/Open-Sora-Plan-v1.0.0/vae/ \                             //vae预训练文件路径
      --data_path dataset/msrvtt/train/annotations.json \                             //数据集配置文件路径
      --video_folder dataset/msrvtt/train/videos.json \                               //视频文件夹路径
      --sample_rate 1 \                                                               //采样率
      --num_frames 17 \                                                               //训练帧数
      --max_image_size 256 \                                                          //图像/视频最大尺寸
      --gradient_checkpointing \                                                      //是否重计算
      --attention_mode math \                                                         //attention的类型
      --train_batch_size=4 \                                                          //训练的批大小
      --dataloader_num_workers 10 \                                                   //数据处理线程数
      --gradient_accumulation_steps=1 \                                               //梯度累计步数
      --max_train_steps=1000000 \                                                     //最大训练步数
      --learning_rate=2e-05 \                                                         //学习率
      --lr_scheduler="constant" \                                                     //学习率调度策略
      --lr_warmup_steps=0 \                                                           //学习率预热步数
      --mixed_precision="bf16" \                                                      //混精训练的数据类型
      --report_to="tensorboard" \                                                     //记录方式
      --checkpointing_steps=2000 \                                                    //检查点步数
      --output_dir="t2v-f17-256-img4-videovae488-bf16-ckpt-xformers-bs4-lr2e-5-t5" \  //输出的路径
      --allow_tf32 \                                                                  //使用tf32训练
      --use_deepspeed \                                                               //使用deepspeed训练
      --model_max_length 300 \                                                        //文本最大长度
      --use_image_num 4 \                                                             //训练使用图片的数量
      --use_img_from_vid                                                              //训练图片来自视频
    ```

#### 性能展示

##### 性能

| 芯片 | 卡数 | 单步迭代时间(s/step)  | batch_size | AMP_Type | Torch_Version |
|:---:|:---:|:----:|:----------:|:---:|:---:|
| GPU | 8p | 1.84 |     4      | bf16 | 2.1 |
| Atlas A2 | 8p | 1.95 |     4      | bf16 | 2.1 |

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
    bash scripts/text_condition/sample_video.sh
    ```

   - 模型在线推理python脚本参数说明如下。

   ```shell
    python opensora/sample/sample_t2v.py \                 //在线推理的Python脚本
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0 \      //LatteT2V预训练权重路径
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \            //文本编码模型权重路径
    --text_prompt examples/prompt_list_0.txt \             //文本提示文件路径
    --ae CausalVAEModel_4x8x8 \                            //视频压缩模型
    --version 65x512x512 \                                 //生成的视频规格
    --save_img_path "./sample_videos/prompt_list_0" \      //生成的视频文件路径
    --fps 24 \                                             //生成视频的帧率
    --guidance_scale 7.5 \                                 //指导尺度
    --num_sampling_steps 250 \                             //采样步数
    --enable_tiling                                        //启用平铺
   ```


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.05.22: VideoGPT bf16训练任务首次发布。

2024.05.30: LatteT2V bf16训练和推理任务首次发布


# FAQ



