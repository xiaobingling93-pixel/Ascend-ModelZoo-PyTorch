# U-ViT for Pytorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [U-ViT(在研版本)](#U-ViT(在研版本))   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [训练任务](#训练任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

U-ViT是一个结合了扩散概率模型和Transformer的图像生成模型，在扩散模型中，使用ViT代替基于CNN的U-Net。
本仓库主要将U-ViT模型迁移到了昇腾NPU上。

## 支持任务列表

本仓已经支持以下模型任务类型


|  模型   | 任务列表  | 是否支持 |
|:-----:|:-----:|:----:|
| U-ViT | train |  ✔   |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/baofff/U-ViT
  commit_id=ce551708dc9cde9818d2af7d84dfadfeb7bd9034
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/diffusion/
  ```

# U-ViT(在研版本)

## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

|     三方库     |  支持版本  |
|:-----------:|:------:|
|   PyTorch   | 1.11.0 |
| accelerate  | 0.12.0 |
| torchvision | 0.12.0 |

  在模型源码包根目录下执行命令，安装模型需要的依赖。

  ```
  pip install -r requirements.txt
  ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

|        软件类型        | 支持版本 |
|:------------------:|:----:|
| FrameworkPTAdapter | 在研版本 |
|        CANN        | 在研版本 |
|      昇腾NPU固件       | 在研版本 | 
|      昇腾NPU驱动       | 在研版本 |



### 准备数据集

用户需自行获取并解压ImageNet数据集，并放在当前目录下assets/datasets/ImageNet文件夹中

参考数据结构如下：

   ```
   assets/datasets/ImageNet
   ├── train
   ├── val
   └── ...
   ```
### 获取训练相关权重

基于[原仓Readme](https://github.com/baofff/U-ViT#preparation-before-training-and-evaluation)中Preparation Before Training and Evaluation一节内提供的链接下载Autoencoder相关文件，并放在当前目录下assets/stable-diffusion文件夹中
   
参考数据结构如下：

   ```
   assets/stable-diffusion
   ├── autoencoder_kl.pth
   └── autoencoder_kl.pth
   ```



基于[原仓Readme](https://github.com/baofff/U-ViT#preparation-before-training-and-evaluation)中Preparation Before Training and Evaluation一节提供的链接下载Reference statistics for FID相关文件，并放在当前目录下assets/fid_stats文件夹中
   
参考数据结构如下：

   ```
   assets/fid_stats
   ├── fid_stats_celeba64_train_50000_ddim.npz
   ├── fid_stats_cifar10_train_pytorch.npz
   ├── fid_stats_imagenet256_guided_diffusion.npz
   ├── fid_stats_imagenet512_guided_diffusion.npz
   ├── fid_stats_imagenet64_guided_diffusion.npz
   └── fid_stats_mscoco256_val.npz
   ```
## 快速开始

### 训练任务

#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡训练。
   
  
   - 单机8卡训练
   
     ```shell
     bash test/train_imagenet64_uvit_large.sh # 8卡训练
     ```
     
   - 训练脚本参数说明如下
      ```shell
      train_imagenet64_uvit_large.sh
      --num_processes           //训练用卡数
      --train_script            //训练脚本
      --config                  //训练参数
      ```
#### 训练结果


##### 性能

|      芯片       | 卡数  |   FPS   | batch_size | AMP_Type | Torch_Version |
|:-------------:|:---:|:-------:|:----------:|:--------:|:-------------:|
|      GPU      | 8p  | 1136.64 |    1024    |   fp16   |     1.13      |
| Atlas 800T A2 | 8p  | 860.50  |    1024    |   fp16   |     1.11      |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.05.06：U-ViT fp16训练任务首次发布。

# FAQ
暂无