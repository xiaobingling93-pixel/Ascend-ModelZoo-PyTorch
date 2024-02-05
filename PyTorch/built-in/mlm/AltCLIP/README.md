# AltCLIP for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

一个简单高效的方法去训练更加优秀的双语CLIP模型。命名为AltCLIP。AltCLIP基于 OpenAI CLIP 训练，训练数据来自 WuDao数据集 和 LAION。训练共有两个阶段。 在平行知识蒸馏阶段，使用平行语料文本来进行蒸馏（平行语料相对于图文对更容易获取且数量更大）。在双语对比学习阶段，使用少量的中-英图像-文本对（一共约2百万）来训练我们的文本编码器以更好地适应图像编码器。

- 参考实现：

  ```
  url=https://github.com/FlagAI-Open/FlagAI
  commit_id=bad326e79a926d700edbc52a82bf8c5cfe43495d
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=https:PyTorch/built-in/mlm
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version |            三方库依赖版本            |
  | :-----------: | :----------------------------------: |
  |  PyTorch 2.1  | deepspeed=0.12.6, torchvision=0.16.2 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。注：先安装以下依赖，再安装torch npu及配套torch包
  ```
  pip install -e .
  pip install -r requirements.txt
  ```
  
- 修改第三方包

  找到当前环境下/path/lib/python3.8/site-packages/transformers/configuration_utils.py文件，搜索logger.info(f"Model config {config}")并注释掉，path为环境安装目录。（官方源码引入，怀疑为第三方库兼容性问题，会引起训练报错）

## 下载预训练模型

- 使用以下命令会自动下载**AltCLIP-XLMR-L**预训练模型、模型配置及CIFAR10数据集，预训练模型、模型配置存放在./checkpoints路径下，如果网络问题无法自动下载，需要在官网手动下载，并放在./checkpoints路径下。

  ```
  bash test/download_dataset.sh
  ```

## 下载数据集

- 脚本会自动下载CIFAR10数据集并存放在./clip_benchmark_datasets目录下，目录结构如下，如果网络问题无法自动下载，需要手动下载，并放在./clip_benchmark_datasets路径下。

   ```
   ├── clip_benchmark_datasets
      ├──cifar-10-batches-py
         ├──batches.meta
         ├──readme.html
         ├──test_batch
         ├──data_batch_1
         ├──data_batch_2
         ...
      ├──cifar-10-python.tar.gz
   ```

# 开始训练

## CIFAR10微调任务

1. 进入解压后的源码包根目录。

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
   

3. 模型训练python训练脚本参数说明如下。

```
altclip_finetuning.py
--batch_size      //设置单卡batch_size
--epoch           //设置epoch数
--lr              //设置学习率
--eval_interval   //设置评估频率
--save_dir        //设置模型存储路径
```

# 训练结果展示

**表 2**  微调任务结果展示表

|   NAME   |  精度  | FPS  | Epochs | precision | batch_size |
| :------: | :----: | :--: | :----: | :-------: | :--------: |
| 8p-竞品A | 0.9737 | 338  |   10   | bfloat16  |    512     |
|  8p-NPU  | 0.9732 | 397  |   10   | bfloat16  |    512     |


# 版本说明

## 变更

2024.02.04：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 ```./public_address_statement.md```