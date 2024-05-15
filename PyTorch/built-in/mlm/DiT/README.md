# DiT for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [DiT](#DiT)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [训练任务](#训练任务) 
       - [在线推理](#在线推理) 
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

Scalable Diffusion Models with Transformers，是完全基于transformer架构的扩散模型，这个工作不仅将transformer成功应用在扩散模型，还探究了transformer架构在扩散模型上的scalability能力，其中最大的模型DiT-XL/2在ImageNet 256x256的类别条件生成上达到了SOTA。

## 支持任务列表
本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| DiT-XL/2 |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/facebookresearch/DiT
  commit_id=ed81ce2229091fd4ecc9a223645f95cf379d582b
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/mlm
    ```

# DiT

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.1.0   |

- 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

  ```shell
  pip install -r requirements.txt
  ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

  **表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.RC2  |
|       CANN        | 8.0.RC2  |
|    昇腾NPU固件    | 24.1.RC2 |
|    昇腾NPU驱动    | 24.1.RC2 |

### 准备预训练权重

- 联网环境下使用以下命令会自动下载**stabilityai/sd-vae-ft-mse**预训练模型。如果网络问题无法自动下载，需要在官网手动下载，存放在任意文件夹中，文件夹内容如下所示，并修改train.py--line174指向上述路径

  ```
  Your-sd-vae-ft-mse-PATH
  ├── config.json
  ├── diffusion_pytorch_model.bin
  ├── diffusion_pytorch_model.safetensors
  ├── README.md
  ```


### 准备数据集

- 自行下载准备imageNet2012数据集，目录结构如下。

```
├── ImageNet2012
      ├──train
           ├──类别1
                 │──图片1
                 │──图片2
                 │   ...       
           ├──类别2
                 │──图片1
                 │──图片2
                 │   ...   
           ├──...                     
      ├──val  
           ├──类别1
                 │──图片1
                 │──图片2
                 │   ...       
           ├──类别2
              │──图片1
                 │──图片2
                 │   ...                
```

> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。      


## 快速开始
### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  1. 进入源码根目录。

     ```
     cd /${模型文件夹名称}
     ```

  2. 运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡训练

     ```
     bash test/train_8p.sh --data_path=/PATH/ImageNet2012/train --image_size=256 --global_batch_size=256 --precision=fp32 --epochs=1
     ```
     
     模型训练脚本参数说明如下。
     
     ```
     train_8p.sh
       --data_path      //训练数据集实际所在路径，请用户根据实际情况修改
       --image_size     //图片大小，支持256和512
       --global_batch_size  //全局batch size设置
       --precision     // 训练精度，支持fp32和bf16
       --epochs        //训练轮数
     ```


#### 训练结果
| 芯片          | 卡数 | image size | global batch size | Precision | 性能FPS |
| ------------- | :--: | :--------: | :---------------: | :-------: | :-----: |
| GPU           |  8p  |    256     |        256        |   fp32    |   432   |
| Atlas 800T A2 |  8p  |    256     |        256        |   fp32    |   376   |
| GPU           |  8p  |    256     |        512        |   bf16    |   727   |
| Atlas 800T A2 |  8p  |    256     |        512        |   bf16    |   586   |
| GPU           |  8p  |    512     |        64         |   fp32    |   80    |
| Atlas 800T A2 |  8p  |    512     |        64         |   fp32    |   77    |
| GPU           |  8p  |    512     |        128        |   bf16    |   151   |
| Atlas 800T A2 |  8p  |    512     |        128        |   bf16    |   122   |

### 在线推理

本任务主要提供**单卡**推理功能。

#### 开始推理

1. 单卡推理命令

```
python sample.py --model DiT-XL/2 --image-size 256 --ckpt /path/to/model.pt
```

脚本入参说明如下。

```
sample.py
  --model      	//模型结构
  --image-size   //图片大小，支持256和512
  --ckpt  		//权重路径，支持官方开源权重和自己训练的权重
```



# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

2024.05.15：首次发布。

# FAQ

无