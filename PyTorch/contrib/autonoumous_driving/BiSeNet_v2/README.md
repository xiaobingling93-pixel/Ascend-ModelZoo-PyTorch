# BiSeNetV2 for PyTorch

# 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [BiSeNetV2](#BiSeNetV2)
  - [准备训练环境](#准备训练环境)
  - [准备数据集](#准备数据集)
  - [快速开始](#快速开始)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)


# 简介

## 模型介绍
语义分割(Semantic Segmentation)核心是低层细节和高层语义。BiSeNetV2将空间细节和分类语义分开处理，在面对实时语义分割时同时实现了高精度和高效率，称为双边分割网络(Bilateral Segment Network, BiSeNetV2)。该架构包括：
细节分支(Detail Branch)：拥有宽的通道和浅的层， 用于捕捉低层细节，并且生成高分辨率的特征表示；
语义分支(Semantic Branch)：拥有窄的通道和宽的层，用于捕捉上下文。

此外，设计了强化训练策略改进分割效果。

本仓主要将BiSeNetV2训练任务迁移到昇腾Atlas A2并做性能优化。

## 代码实现
- 参考实现：
  ```text
  url=https://github.com/CoinCheung/BiSeNet
  commit_id=f2b901599752ce50656d2e50908acecd06f7eb47
  ```

- 适配昇腾AI处理器的实现
  ```text
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/autonoumous_driving
  ```

# BiSeNetV2
## 准备训练环境
### 安装算法环境
- 当前模型支持的PyTorch版本依赖如下表所示
  
    表1 三方库版本支持表
    | 依赖 | 版本 |
    | :---: | :---: |
    | Python | 3.9.x |
    | PyTorch | 1.11.0 |
    | tabulate | 0.9.x |

- 当前模型在以下昇腾环境测试验证如下表所示

    表2 昇腾软件版本支持表
    | 软件类型 | 支持版本 |
    | :---: | :---: |
    | FrameworkPTAdapter | 6.0.RC1 |
    | CANN | 8.0.RC1 |
    | Ascend HDK | 24.1.RC1 |
    | PyTorch | 1.11.0 |

- 环境准备指导

  请参考《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖
    
    在模型源码包目录下执行命令，安装模型对应PyTorch版本需要的依赖。
    ```bash
    pip install -r 1.11_requirements.txt  # PyTorch 1.11版本
    ```

- 编译OpenCV（可选）

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

## 数据准备

### 获取数据集

用户自行获取原始数据集，以cityscapes数据集为例。

cityscapes数据集下载地址：https://www.cityscapes-dataset.com/

```bash
cd /${模型文件夹名称}
mkdir -p datasets/cityscapes

unzip /${数据集下载路径}/leftImg8bit_trainvaltest.zip datasets/cityscapes
unzip /${数据集下载路径}/gtFine_trainvaltest.zip datasets/cityscapes
```

拷贝训练/验证索引文件
```bash
cd /${模型文件夹名称}
# train.txt和val.txt可从原工程https://github.com/CoinCheung/BiSeNet datasets/cityscapes下获取。
cp /${索引下载路径}/train.txt datasets/cityscapes
cp /${索引下载路径}/val.txt datasets/cityscapes
```

数据集目录结构如下：
```text
.
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── train.txt
└── val.txt
```

> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。


### 获取预训练模型

若用户网络支持训练时动态下载可跳过该节

用户自行获取预训练模型，以backbone_v2.pth为例，放置到算法运行环境(或容器)的主目录下的hub/checkpoints文件夹内。
```bash
mkdir -p ${HOME}/.cache/torch/hub/checkpoints
wget https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth -O ${HOME}/.cache/torch/hub/checkpoints/
```

## 快速开始
### 训练模型
1、进入解压后的源码包根目录  
```bash
cd /${模型文件夹名称}
```

2、运行训练脚本
- 单机单卡训练
    ```bash
    # 单卡精度测试脚本
    bash ./test/train_full_1p.sh
    # 单卡性能测试脚本
    bash ./test/train_performance_1p.sh
    ```
- 单机8卡训练
    ```bash
    # 8卡精度测试脚本
    bash ./test/train_full_8p.sh
    # 8卡性能测试脚本
    bash ./test/train_performance_8p.sh
    ```

### 训练结果
表3：mious对比图 batchsize=16, num_workers=32

| 芯片 | ss | ssc | msf | msfc |
| :---: | :---: | :---: | :---: | :---: |
| 竞品A | 0.725 | 0.737 | 0.745 | 0.757 |
| Atlas 800T A2 | 0.724 | 0.734 | 0.746 | 0.755 |

# 公网地址说明
代码涉及公网地址参考 [public_address_statement.md](public_address_statement.md)

# 变更说明
2024.04.01: 适配BiSeNetV2 Atlas A2性能优化。

2024.04.11: 更新readme.md。

# FAQ

无