# BEVDet_for_PyTorch

# 目录
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [预训练数据集](#预训练数据集)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [训练结果](#训练结果)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)


# 简介

## 模型介绍

*BEVDet*是一种用于3D目标检测的深度学习模型，可以从一个俯视图像中检测出三维空间中的物体，并预测他们的位置、大小和朝向。在自动驾驶、智能交通等领域中有广泛应用。其基于深度学习技术，使用卷积神经网络和残差网络，在训练过程中使用了大量的3D边界框数据，以优化模型的性能和准确性。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/HuangJunJie2017/BEVDet.git
  commit_id=58c2587a8f89a1927926f0bdb6cde2917c91a9a5
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/autonoumous_driving
  ```

# 准备训练环境
## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 8.0.RC2  |
  |       CANN         | 8.0.RC2  |
  |      昇腾NPU固件       | 24.0.RC2 |
  |      昇腾NPU驱动       | 24.0.RC2 |

## 安装模型环境

 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库       |  支持版本  |
  |:--------------:|:------:|
  |    PyTorch     |  2.1   |
  |    ADS-Accelerator     | latest |
  |      mmcv      |  1.x   |
  |     mmdet      | 2.28.2 |
  | mmsegmentation | 0.30.0 |

- 安装ADS-Accelerator

  请参考昇腾[ads](https://gitee.com/ascend/ads)代码仓说明编译安装ADS-Accelerator

- 安装基础依赖

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  
  ```
  pip install opencv-python==4.9.0.80

  pip install -r requirements.txt
  ```

- 安装mmcv

  在mmcv官网获取[mmcv 1.x](https://github.com/open-mmlab/mmcv/tree/1.x)分支源码，解压至`$YOURMMCVPATH`。将`mmcv_replace`中的文件拷贝到`$YOURMMCVPATH/mmcv`覆盖原文件。运行以下命令
  ```
  cd $YOURMMCVPATH
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```
- 安装mmdet和mmsegmentation
  ```
  pip install mmdet==2.28.2
  pip install mmsegmentation==0.30.0
  ```

# 准备数据集

## 预训练数据集
用户自行获取*nuscenes*数据集，在源码目录创建软连接`data/nuscenes`指向解压后的nuscenes数据目录

运行数据预处理脚本生成BEVDet模型训练需要的pkl文件
  ```
  python tools/create_data_bevdet.py
  
  ```

  整理好的数据集目录如下:

```
BEVDet_for_PyTorch/data
    nuscenes
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
        nuscenes_infos_train.pkl
        nuscenes_infos_val.pkl
        bevdetv3-nuscenes_infos_train.pkl
        bevdetv3-nuscenes_infos_val.pkl
```

# 快速开始

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. *lidar segmentation dim64*任务训练

- 单机单卡训练

     ```
     bash ./test/train_1p.sh --py_config=config/bevdet/bevdet-r50.py # 单卡精度
     
     bash ./test/train_1p.sh --py_config=config/bevdet/bevdet-r50 --performance=1  # 单卡性能
     ```
   
- 单机8卡训练

     ```
     bash ./test/train_8p.sh --py_config=config/bevdet/bevdet-r50 # 8卡精度

     bash ./test/train_8p.sh --py_config=config/bevdet/bevdet-r50 --performance=1 # 8卡性能 
     ```

  模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --py_config                              //不同类型任务配置文件
   --performance                            //--performance=1开启性能测试，默认不开启
   --work_dir                               //输出路径包括日志和训练参数
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


## 训练结果

**表 3** 训练结果展示表

|  芯片      | 卡数 | mAP  | FPS  | Max epochs |
|:--------:|----|:----:|:----:|:----------:|
|   GPU    | 1p |  -   | 1.53 |     1      |
|   GPU    | 8p | 28.3 | 9.64 |     24     |
| Atlas A2 | 1p |  -   | 0.61 |     1      |
| Atlas A2 | 8p | 28.1 | 3.67 |     24     |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明
2024.05.27：首次发布。

## FAQ
暂无。



