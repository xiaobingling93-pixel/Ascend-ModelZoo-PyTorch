# Open-Sora-Plan v1.0 for Pytorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [VideoGPT](#VideoGPT)   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [训练任务](#训练任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

Open-Sora-Plan是由北大技术团队推出的项目，旨在通过开源框架复现 OpenAI Sora。作为基础开源框架，它支持视频生成模型的训练，包括无条件视频生成、课堂视频生成和文本到视频生成
本仓库主要将Open-Sora-Plan多个任务迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|  模型  |    任务列表    | 是否支持 |
|:----:|:----------:|:-----:|
| VideoGPT |    训练     | ✔ |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/PKU-YuanGroup/Open-Sora-Plan
  commit_id=a7375034586fea20b4aa14bc17c58adbaeeef32f
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mlm/OpenSoraPlan1.0
  ```


# VideoGPT

## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  | 三方库    | 支持版本  |
  | :--------: | :-------------: |
  | PyTorch | 2.1.0 |
  | diffusers | 0.27.2 |
  | accelerate | 0.28.0 | 
  | deepspeed | 0.12.6 |
  | transformers | 4.39.1 |
  | decord | 0.6.0 |


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```python
   pip install -e .                                                # 安装本地OpenSoraPlan代码仓     
   ```
   注: 模型依赖decord需编译安装，根据原仓安装https://github.com/dmlc/decord
### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

  | 软件类型   | 支持版本  |
  | :--------: | :-------------: |
  | FrameworkPTAdapter | 6.0.RC2 |
  | CANN | 8.0.RC2 |
  | 昇腾NPU固件 | 24.1.RC2 | 
  | 昇腾NPU驱动 | 24.1.RC2 |

  

### 准备数据集

#### 训练数据集准备

1. 用户需自行获取并解压MSRVTT数据集，并在以下启动shell脚本中将`data_path`参数设置为本地数据集的绝对路径。

   ```shell
   ./scripts/videogpt/train_videogpt.sh
   ```

   数据结构如下：

   ```
   $LAION5B
   ├── annotation
   ├── high-quality
   ├── structured-symlinks
   └── video
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
     bash ./scripts/videogpt/train_videogpt.sh # 8卡训练，混精bf16
     ```
      
   - 模型训练python训练脚本参数说明如下。
   
   ```shell
   ./scripts/videogpt/train_videogpt.sh
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
   --save_steps                        //保存步数
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
   




# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.05.22：VideoGPT bf16训练任务首次发布。


# FAQ



