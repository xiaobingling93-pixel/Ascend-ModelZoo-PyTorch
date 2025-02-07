# Resnet50 for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

ResNet是由微软研究院的Kaiming He等四名华人提出，是ImageNet竞赛中分类问题效果较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练，可以极快的加速神经网络的训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。ResNet18的含义是指网络中有18-layer。本文档描述的ResNet50是基于Pytorch实现的版本。


- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=e6cba0aa46b2a33b01207e1451e0cd10ca96c04c
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```


# 准备训练环境

该模型为随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），您可以根据下面提供的安装指导选择匹配的CANN等软件下载使用。

## 准备环境

- 推荐参考[配套资源文档](https://www.hiascend.com/developer/download/commercial)使用最新的配套版本。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 24.1.RC3 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 24.1.RC3 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.0.RC3 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v6.0.rc3 </td>
    </tr>
  </table>

- 三方库依赖如下表所示。

  **表 2**  三方库依赖表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 2.1 | pillow==9.1.0 |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```

  pip install -r 2.1_requirements.txt  # PyTorch2.1版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   Resnet18迁移使用到的ImageNet2012数据集目录结构参考如下所示。

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
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

  - 单机16卡训练

     启动16卡训练。

     ```
     bash ./test/train_full_16p.sh --data_path=/data/xxx/  # 16卡精度
     
     bash ./test/train_performance_16p.sh --data_path=/data/xxx/  # 16卡性能
     ```

   - 多机多卡训练

     启动多机多卡训练。

     ```
     bash ./test/train_cluster.sh --data_path=xxx --batch_size="xxx" --lr="xxx" --train_epochs="xxx" --world_size="xxx" --node_rank="xxx" --master_addr="xxx"
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录；
   
   --batch_size网络训练的batch size，集群bs的设置推荐: 总卡数 * 512；
   
   --train_epochs网络训练周期；
   
   --world_size集群训练节点数；
   
   --node_rank集群训练节点ID，每个节点不一样；
   
   --master_addr集群训练主节点ip；
   
   --lr集群训练学习率，4机32卡训练学习率推荐四点多。

   --hf32开启HF32模式，不与FP32模式同时开启

   --fp32开启FP32模式，不与HF32模式同时开启

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --arch                              //使用模型，默认：resnet50
   --workers                           //加载数据进程数，默认：4
   --epochs                            //重复训练次数，默认90
   --batch-size                        //训练批次大小
   --lr                                //学习率，默认0.2
   --world-size                        //分布式训练节点数
   --rank                              //进程编号，默认：-1
   --seed                              //使用随机数种子
   --amp                               //是否使用混合精度
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

| NAME  | Acc@1 |  FPS  | Epochs | AMP_Type  | Torch_Version |
|:-:|:-----:|:-----:|:------:|:-:|:-:|
| 1p-NPU |   -   | 1678  |   1    |    O2     |     1.11      |
| 8p-NPU | 76.3  | 13212 |   90   |    O2     |     1.11      |
| 1p-NPU |   -   | 1678  |   1    |    O2     |      2.1      |
| 8p-NPU | 76.3  | 13255 |   90   |    O2     |      2.1      |
| 16p-NPU  | 76.69 | 30000  |   90   | O2  | 2.1  |

说明：上表为历史数据，仅供参考。2024年12月31日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 8p-竞品 | FP16 | 13255 |
| 8p-Atlas 900 A2 PoDc | FP16 | 14321 |

# 版本说明

## 变更

2024.06.28: 新增单机16卡脚本，增加16卡性能基线。

2023.02.24：更新readme，重新发布。

2022.12.14：首次发布。

## FAQ

无。
