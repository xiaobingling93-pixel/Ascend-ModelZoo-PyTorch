# SSD-MobileNetV1 for PyTorch 

# 目录

-   [简介](#简介)
    -   [模型介绍](#模型介绍)
    -   [代码实现](#代码实现)
-   [SSD-MobileNetV1](#ssd-mobilenetv1)
    -   [准备训练环境](#准备训练环境)
    -   [准备数据集](#准备数据集)
    -   [快速开始](#快速开始)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#faq)


# 简介

## 模型介绍

MobileNetV1是基于深度级可分离卷积构建的网络。 MobileNetV1将标准卷积拆分为了两个操作：深度卷积和逐点卷积 。
SSD是一种one-stage的目标检测框架。SSD_MobileNetV1使用MobileNetV1提取有效特征，之后SSD通过得到的特征图的信息进行检测。
本仓主要将SSD_MobileNetV1训练任务迁移到了昇腾NPU上，并进行极致性能优化。

## 代码实现

- 参考实现：

  ```
  url=https://github.com/qfgaohao/pytorch-ssd
  commit_id=7a839cbc8c3fb39679856b4dc42a1ab19ec07581
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/autonoumous_driving
  ```


# SSD-MobileNetV1
## 准备训练环境

### 安装模型环境

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
      <td> AscendHDK 24.1.RC2.3 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 24.1.RC2.3 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.0.RC2.2 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 1.11.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v6.0.rc2 </td>
    </tr>
  </table>

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.11_requirements.txt
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。

- 源码构建OpenCV

  来源：
  ```
  https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000054.html
  ```

  安装前准备：
  ```
  export GIT_SSL_NO_VERIFY=true
  ```

  下载源码包：
  ```
  git clone https://github.com/opencv/opencv.git

  cd opencv
  
  mkdir -p build
  ```

  执行命令：
  ```
  cd build  

  cmake -D BUILD_opencv_python3=yes -D BUILD_opencv_python2=no -D PYTHON3_EXECUTABLE=/home/ma-user/anaconda3/env/PyTorch-1.11.0/bin/python -D PYTHON3_INCLUDE_DIR=/home/ma-user/anaconda3/env/PyTorch-1.11.0/include/python3.9 -D
  PYTHON3_LIBRARY=/home/ma-user/anaconda3/env/PyTorch-1.11.0/lib/libpython3.9.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/ma-user/anaconda3/env/PyTorch-1.11.0/lib/python3.9/site-packages/numpy/core/include -D     
  PYTHON3_PACKAGES_PATH=/home/ma-user/anaconda3/env/PyTorch-1.11.0/lib/python3.9/site-packages -D PYTHON3_DEFAULT_EXECUTABLE=/home/ma-user/anaconda3/env/PyTorch-1.11.0/bin/python ..  

  make -j$nproc
  
  make install
  ```

## 准备数据集

### 获取数据集

   用户自行获取原始数据集，可选用的开源数据集包括VOCdevkit等，将数据集上传到服务器任意路径下并解压。

   以VOCdevkit数据集为例，数据集目录结构参考如下所示。

   ```
   |——VOCdevkit
        |——VOC2007（VOC2012）
            |——Annotations
            |——ImageSets
            |——JPEGImages
            |——SegmentationClass
            |——SegmentationObject
        |——test
            |——VOC2007
                |——Annotations
                |——ImageSets
                |——JPEGImages
                |——SegmentationClass
                |——SegmentationObject
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

### 获取预训练模型

请用户在源码包根目录下新建"models/"文件夹，下载所需的预训练模型**mobilenet_v1_with_relu_69_5.pth**，并将预训练模型放置在"models/"文件夹下。

模型下载方法链接：https://github.com/qfgaohao/pytorch-ssd/blob/master/README.md#download-models


## 快速开始

### 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。
   以下命令为数据集在代码目录下的样例，
   也可以自行添加软链接到数据集位置。

   - 单机单卡训练


     ```
     bash ./test/train_full_1p.sh --data_path=VOCdevkit/VOC2007/ --validation_data_path=VOCdevkit/test/VOC2007/  # 单卡精度

     bash ./test/train_full_1p.sh --data_path=VOCdevkit/VOC2007/,VOCdevkit/VOC2012/ --validation_data_path=VOCdevkit/test/VOC2007/  # 单卡多数据集精度
     
     bash ./test/train_performance_1p.sh --data_path=VOCdevkit/VOC2007/ --validation_data_path=VOCdevkit/test/VOC2007/  # 单卡性能

     bash ./test/train_performance_1p.sh --data_path=VOCdevkit/VOC2007/,VOCdevkit/VOC2012/ --validation_data_path=VOCdevkit/test/VOC2007/  # 单卡多数据集性能
     ```

   - 单机8卡训练

     ```
     bash ./test/train_full_8p.sh --data_path=VOCdevkit/VOC2007/ --validation_data_path=VOCdevkit/test/VOC2007/  # 8卡精度

     bash ./test/train_full_8p.sh --data_path=VOCdevkit/VOC2007/,VOCdevkit/VOC2012/ --validation_data_path=VOCdevkit/test/VOC2007/  # 8卡多数据集精度
     
     bash ./test/train_performance_8p.sh --data_path=VOCdevkit/VOC2007/ --validation_data_path=VOCdevkit/test/VOC2007/  # 8卡性能

     bash ./test/train_performance_8p.sh --data_path=VOCdevkit/VOC2007/,VOCdevkit/VOC2012/ --validation_data_path=VOCdevkit/test/VOC2007/  # 8卡多数据集性能
     ```

      --data_path 参数填写数据集路径，需写到数据集的一级目录；

      --validation_data_path 参数填写测试集路径，需写到数据集的一级目录。

   - 评测

     ```
     bash ./test/train_eval.sh --data_path=VOCdevkit/test/VOC2007/ --pth_path=models/mb1-ssd-Epoch-xxx-Loss-xxxxxxxx.pth
     ```
      --data_path 参数填写测试集路径，需写到测试集一级目录；
   
      --pth_path 参数填写训练过程中生成的权重文件路径（默认存储在"models/"文件夹下）。

### 训练结果

**表 2** 训练结果展示表

| NAME	       |  卡数  | 	mAP   |  FPS  |    batch_size    |   Epochs  |Torch_Version|
|-------------|--------|----------|-------|------------|-----------|--------|
|  Atlas 900 A2 PODc	 |    8    |0.6849	  | 6250.54	    |     32     | 240	     | 1.11   |
|  竞品A  |     8    |0.657     | 6547      |      32      | 240      | 1.11   |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

2024.03.25：优化模型的训练新能。

2024.03.26：更新readme.md。

# FAQ

无。