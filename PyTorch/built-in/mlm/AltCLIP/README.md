# AltCLIP for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [AltCLIP](#AltCLIP)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [CIFAR10微调任务](#CIFAR10微调任务) 
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

提出了一个简单高效的方法去训练更加优秀的双语CLIP模型。命名为AltCLIP。AltCLIP基于OpenAI CLIP训练，训练数据来自 WuDao和 LAION数据集。训练共有两个阶段。在平行知识蒸馏阶段，使用平行语料文本来进行蒸馏（平行语料相对于图文对更容易获取且数量更大）。在双语对比学习阶段，使用少量的中-英图像-文本对（一共约2百万）来训练我们的文本编码器以更好地适应图像编码器。

## 支持任务列表
本仓已经支持以下模型任务类型

|      模型      | 任务列表 | 是否支持 |
| :------------: | :------: | :------: |
| AltCLIP-XLMR-L |   微调   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/FlagAI-Open/FlagAI
  commit_id=bad326e79a926d700edbc52a82bf8c5cfe43495d
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/mlm
    ```

# AltCLIP

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

|   三方库    | 支持版本 |
| :---------: | :------: |
|   PyTorch   |  2.1.0   |
| TorchVision |  0.16.0  |
|  deepspeed  |  0.12.6  |

- 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。注：由于某些依赖会自动安装torch包，因此需要先安装以下依赖，再手动安装torch npu及配套torch包进行覆盖。



  ```shell
  pip install -e .
  pip install -r requirements.txt
  ```

- 修改第三方包


找到当前环境下/path/lib/python3.8/site-packages/transformers/configuration_utils.py文件，搜索logger.info(f"Model config {config}")并注释掉，path为环境安装目录。（官方源码引入，怀疑为第三方库兼容性问题，会引起训练报错）

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

  **表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.RC1  |
|       CANN        | 8.0.RC1  |
|    昇腾NPU固件    | 24.1.RC1 |
|    昇腾NPU驱动    | 24.1.RC1 |

### 准备预训练权重

- 联网环境下使用以下命令会自动下载**AltCLIP-XLMR-L**预训练模型、模型配置及CIFAR10数据集，预训练模型、模型配置存放在./checkpoints路径下，如果网络问题无法自动下载，需要在官网手动下载，并放在./checkpoints路径下。

  ```
  bash test/download_dataset.sh
  ```

- 离线环境下请前往原仓手动下载预训练模型及相关配置并存放在./checkpoints路径下，目录结构如下所示。由于源码离线加载模型bug，需要在./flagai/model/base_model.py-262与263行之间插入代码"model_id = None"。

  ```
  checkpoints
  ├── AltCLIP-XLMR-L
      ├── config.json
      ├── preprocessor_config.json
      ├── pytorch_model.bin
      ├── special_tokens_map.json
      ├── tokenizer_config.json
      ├── tokenizer.json
  ```

  

### 准备数据集

- 上节提到的脚本会自动下载CIFAR10数据集并存放在./clip_benchmark_datasets目录下，离线环境需要手动下载，并放在./clip_benchmark_datasets路径下，目录结构如下。

```
clip_benchmark_datasets
├──cifar10
   ├──cifar-10-batches-py
      ├──batches.meta
      ├──readme.html
      ├──test_batch
      ├──data_batch_1
      ├──data_batch_2
      ├──...
   ├──cifar-10-python.tar.gz
```

> **说明：**  
>  该数据集的训练过程脚本只作为一种参考示例。      


## 快速开始
### CIFAR10微调任务

本任务主要提供**bp16**的**8卡**训练脚本。

#### 开始训练

  1. 进入源码根目录。

     ```
     cd /${模型文件夹名称}
     ```

  2. 运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡训练

     ```
     bash test/train_full_8p.sh             # 8卡训练
     bash test/train_performance_8p.sh      # 8卡性能
     ```

     模型训练脚本参数说明如下。

     ```
     altclip_finetuning.py
       --batch_size      //设置单卡batch_size
       --epoch           //设置epoch数
       --lr              //设置学习率
       --eval_interval   //设置评估频率
       --save_dir        //设置模型存储路径
     ```


#### 训练结果
| 芯片                 | 卡数 | 精度acc | 性能FPS | batch size | Precision | Torch Version |
| -------------------- | :--: | :-----: | :-----: | :--------: | :-------: | :-----------: |
| GPU                  |  8p  | 0.9737  |   338   |    512     |   bf16    |      2.1      |
| Atlas A200T A2 Box16 |  8p  | 0.9732  |   295   |    512     |   bf16    |      2.1      |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md 及 README.md

# 变更说明

2024.02.04：首次发布。

2024.03.11：README整改。

# FAQ

无