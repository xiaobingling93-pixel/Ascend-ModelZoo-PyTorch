# EfficientNetV2 for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)



# 概述

## 简述

EfficientNetV2是Efficient的改进版，accuracy达到了发布时的SOTA水平，而且训练速度更快参数量更少。相对EfficientNetV1系列只关注准确率，参数量以及FLOPs，V2版本更加关注模型的实际训练速度。


- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models/blob/v0.5.4/timm/models/efficientnet.py
  commit_id=9ca343717826578b0e003e78b694361621c2b0ef
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```


# 准备训练环境

该模型为随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），您可以根据下面提供的安装指导选择匹配的CANN等软件下载使用。

## 准备环境

- 推荐使用最新的版本准备训练环境。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 25.0.RC1.1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.0.RC1.1 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.1.RC1 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v7.0.0-pytorch2.1.0 </td>
    </tr>
  </table>

- 三方库依赖如下表所示。

  **表 2**  三方库依赖表

  | Torch_Version |    三方库依赖版本    |
  |:-------------:|:-------------:|
  |  PyTorch 2.1  | pillow==9.1.0 |
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r requirements.txt
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   EfficientNetV2迁移使用到的ImageNet2012数据集目录结构参考如下所示。

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

--data_path参数填写数据集路径，需写到数据集的一级目录。

模型训练脚本参数说明如下。

   ```
公共参数：
--momentum                           //动量
--weight-decay                       //权重衰减
--lr                                 //初始学习率
--epochs                             //训练周期数
--seed                               //随机数种子设置
--workers                            //训练进程数
--model                              //训练模型名
--batch-size                         //训练批次大小
--apex-amp                           //使用apex进行混合精度训练（默认）
--native-amp                         //使用amp进行混合精度训练，启用时须将--apex-amp删除
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

|   NAME   | Acc@1 |   FPS   | Epochs | AMP_Type | Torch_Version |
| :------: |:-----:|:-------:|:------:| :------: | :-----------: |
| 1p-竞品V |   -   | 533  |   1    |    O1    |      1.8      |
| 8p-竞品V | 82.34 | 4100 |  350   |    O1    |      1.8      |
| 1p-NPU |   -   | 1110.2  |   1    |    O1     |     1.11      |
| 8p-NPU | 82.19 | 6879.25 |  350   |    O1    |     1.11      |
| 1p-NPU |   -   | 1100.73 |   1    |    O1     |      2.1      |
| 8p-NPU | 82.19 | 6914.44 |  350   |    O1     |      2.1      |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 8p-竞品 | FP16 | 6914.44 |
| 8p-Atlas 900 A2 PoDc | FP16 | 7603.85 |

# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

2022.10.14：首次发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md