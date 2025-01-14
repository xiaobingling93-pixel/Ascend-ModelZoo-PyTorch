# 当前仓已不维护，请跳转至 https://gitee.com/ascend/DrivingSDK/tree/master/model_examples/SurroundOcc

# SurroundOcc for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [SurroundOcc](#SurroundOcc)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [训练任务](#训练任务) 
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

传统的 3D 场景理解方法大多数都集中在 3D 目标检测上，难以描述任意形状和无限类别的真实世界物体。而 SurroundOcc 方法可以更全面地感知 3D 场景。首先对每个图像提取多尺度特征，并采用空间 2D-3D 注意力将其提升到 3D 体积空间；然后，采用 3D 卷积逐步上采样体积特征，并在多个级别上施加监督。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| SurroundOcc |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/weiyithu/SurroundOcc
  commit_id=f698d7968a60815067601776dabfca2a8b03500a
  ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/autonoumous_driving 
    ```

# SurroundOcc

## 准备训练环境

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

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装mmdet3d

  - 在模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    ```

  - 在mmdetection3d目录下，修改代码

    （1）删除requirements/runtime.txt中第3行 numba==0.53.0

    （2）修改mmdet3d/____init____.py中第22行 mmcv_maximum_version = '1.7.0'为mmcv_maximum_version = '1.7.2'

  - 安装包

    ```
    pip install -v -e .
    ```

- 安装mmcv

  - 在模型根目录下，克隆mmcv仓，并进入mmcv目录安装

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e . -v
    ```
- 安装mx_driving
  - 参考mxDriving官方gitee仓README安装编译构建并安装mxDriving包：[参考链接](https://gitee.com/ascend/mxDriving)
  【注意】当前版本配套mxDriving RC3及以上版本，历史mxDriving版本需要model仓代码回退到git reset --hard 91ac141ecfe5872f4835eef6aa4662f46ede80c3
  
- 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

  ```
  pip install -r requirements.txt
  ```

- 在当前python环境下执行'pip show pip'，得到三方包安装路径Location，记作location_path，在模型根目录下执行以下命令来替换patch。

  ```
  bash replace_patch.sh --packages_path=location_path
  ```

- 安装mxDriving加速库，安装方法参考[原仓](https://gitee.com/ascend/mxDriving)，安装后手动source环境变量或将其配置在test/env_npu.sh中。


### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
SurroundOcc
├── data/
│   ├── nuscenes/
│   ├── nuscenes_occ/
│   ├── nuscenes_infos_train.pkl
│   ├── nuscenes_infos_val.pkl
```

> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。      

### 准备预训练权重

- 根据原仓Installation章节下载预训练权重r101_dcn_fcos3d_pretrain.pth，并放在模型根目录ckpts下。

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  1. 在模型根目录下，运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡精度训练

     ```
     bash test/train_8p.sh
     ```

     - 单机8卡性能训练


     ```
     bash test/train_8p_performance.sh
     ```


#### 训练结果
| 芯片          | 卡数 | global batch size | Precision | epoch |  IoU   |  mIoU  | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :----: | :-------------------: |
| 竞品A           |  8p  |         8         |   fp32    |  12   | 0.3163 | 0.1999 |         1028          |
| Atlas 800T A2 |  8p  |         8         |   fp32    |  12   | 0.3169 | 0.1985 |         1654          |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

2024.05.30：首次发布。

# FAQ

无