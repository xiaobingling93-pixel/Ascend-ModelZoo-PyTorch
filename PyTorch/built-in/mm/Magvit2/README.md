# Magvit2 for PyTorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [Magvit2](#Magvit2)   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [训练任务](#训练任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

Magvit2是谷歌提出的利用语言模型结构为视频和图像生成简洁且富有表现力的编码模型。
本仓库主要将Magvit2模型迁移到了昇腾NPU上。

## 支持任务列表

本仓已经支持以下模型任务类型


|   模型    | 任务列表  | 是否支持 |
|:-------:|:-----:|:----:|
| magvit2 | train |  ✔   |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/lucidrains/magvit2-pytorch
  commit_id=06edbd31a4daf3468f2af96bafdb5ef1b0259b19
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/mm/
  ```

# Magvit2

## 准备训练环境

### 安装模型环境

  该模型需要python3.10及以上版本，支持的PyTorch版本和三方库依赖如下表所示  

  **表 1**  三方库版本支持表

|     三方库     |  支持版本  |
|:-----------:|:------:|
|   PyTorch   | 2.1.0  |
| torchvision | 0.16.0 |

  在模型源码包根目录下执行命令，安装模型需要的依赖。

  ```
  pip install -e .
  ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com /document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

|        软件类型        |   支持版本   |
|:------------------:|:--------:|
| FrameworkPTAdapter | 在研版本  |
|        CANN        | 在研版本  |
|      昇腾NPU固件       | 在研版本 | 
|      昇腾NPU驱动       |   在研版本   |


### 准备数据集

用户需自行获取并解压MSRVTT数据集 

参考数据结构如下：

   ```
   /path/to/dataset/MSRVTT/videos/all/
   ├── video0.mp4
   ├── video1.mp4
   └── ...
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
     bash test/train_full_8p_magvit2.sh --dataset_folder=/path/to/dataset/MSRVTT/videos/all/ # 8卡训练
     bash test/train_perf_8p_magvit2.sh --dataset_folder=/path/to/dataset/MSRVTT/videos/all/ # 8卡性能
     ```
     
- 训练参数如下
   ```shell
     dataset_folder                       \\训练数据所在文件夹
     batch_size                           \\训练batchsize
     grad_accum_every                     \\梯度累积次数
     learning_rate                        \\学习率 
     num_train_steps                      \\最大训练步数
   ```
#### 训练结果


##### 性能

|        芯片         | 卡数  | 单步迭代时间（s/step) | batch_size | AMP_Type | Torch_Version |
|:-----------------:|:---:|:--------------:|:----------:|:--------:|:-------------:|
|        竞品A        | 8p  |      1.12      |     16     |   bf16   |      2.1      |
| Atlas 800T A2 | 8p  |      1.40      |     16     |   bf16   |      2.1      |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.08.31：Magvit2 bf16训练任务首次发布。

# FAQ
暂无