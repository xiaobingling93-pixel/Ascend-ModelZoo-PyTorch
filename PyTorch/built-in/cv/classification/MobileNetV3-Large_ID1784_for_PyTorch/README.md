# MobileNetv3 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述
## 简述

MobileNetV3结合了MobileNetV1的深度可分离卷积、MobileNetV2的Inverted Residuals和Linear Bottleneck、SE模块，利用NAS（神经结构搜索）来搜索网络的配置和参数，采用了新的非线性激活层h-swish，在精度和性能方面较MobileNetV2均有一定程度提高。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples/tree/main/imagenet
  commit_id=f5bb60f8e6b2881be3a2ea8c9a3d43e676aa2340
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
  | PyTorch 2.1 | torchvision==0.16.0 |

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

   用户自行下载 `ImageNet` 数据集，将数据集上传到服务器任意路径下并解压。
   
   数据集目录结构参考如下所示。

   ```
   ├── ImageNet
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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    # 单卡精度
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/ # 8卡性能
     ```
   - 在线推理

     启动在线推理。

     ```
     bash ./test/eval.sh --data_path=/data/xxx/ --resume=/resume_path  # 在线推理
     ```

    --data_path参数填写数据集路径，需写到数据集的一级目录，--resume参数填写模型权重

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --arch                         //模型名称
   --data                         //数据集路径
   --batch_size                   //训练批次大小
   --learning-rate                //初始学习率
   --print-freq                   //打印频率
   --epochs                       //重复训练次数
   --apex                         //使用混合精度
   --apex-opt-level               // apex优化器级别
   --lr-step-size                 // 学习率调整步数大小
   --lr-gamma                     // lr 伽马参数
   --wd                           // 权重衰减参数
   --device_id                     //使用设备
   --workers                      //工作线程数
   --max_steps                    // 性能模型最大执行步数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 3**  训练结果展示表

| NAME    | Acc@1  |   FPS   | Epochs | AMP_Type | Torch_Version |
| :-----: | :----: |:-------:|:------:| :------: |  :-------:    |
| 1p-竞品V| - | - | 1  | O2 | 1.5 |
| 8p-竞品V| 74.0 | 3885   | 600    |    O2    | 1.5 |
| 1p-NPU | - | 1916.25 |   1    |     O2      |     1.11      |
| 8p-NPU | 73.5 | 9894.58 |  600   |     O2      |     1.11      |
| 1p-NPU | - | 1825.02 |   1    |     O2      |      2.1      |
| 8p-NPU | 73.5 | 9451.71 |  600   |     O2      |      2.1      |

说明：上表为历史数据，仅供参考。2024年12月31日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 8p-竞品 | FP16 | 9451.71 |
| 8p-Atlas 900 A2 PoDc | FP16 | 14462.5 |

# 版本说明

## 变更
2023.03.22：更新模型精度基线

2022.10.24：更新torch1.8版本，重新发布。

2021.07.05：首次发布。

## FAQ

1. 在ARM平台上，安装0.6.0版本的torchvision，需进行源码编译安装，可以参考源码readme进行安装。
   
   ```
   https://github.com/pytorch/vision
   ```
   
# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
