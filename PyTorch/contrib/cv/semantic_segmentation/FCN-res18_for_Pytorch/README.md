# FCN-res18_for_Pytorch

# 目录
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [预训练数据集](#预训练数据集)
  - [获取预训练权重](#获取预训练权重)
- [快速开始](#快速开始)
  - [训练模型](#训练模型)
  - [训练结果](#训练结果)
- [公网地址说明](#公网地址说明)
- [变更说明](#变更说明)
- [FAQ](#FAQ)


# 简介


## 模型介绍

*FCN-res18*是一个经典的语义分割网络，*FCN-res18*使用全卷积结构，可以接受任意尺寸的输入图像，采用反卷积对最后一层的特征图进行上采样，得到与输入图像相同尺寸的输出，从而对输入进行逐像素预测。

## 代码实现
- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  commit_id=c685fe6767c4cadf6b051983ca6208f1b9d1ccb8
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/semantic_segmentation
  ```
  注：请将参考实现原仓中*mmseg/utils*目录下的*bpe_simple_vocab_16e6.txt.gz*文件手动下载后放置到适配实现的同目录下

# 准备训练环境
## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 6.0.RC1  |
  |       CANN         | 8.0.RC1  |
  |      昇腾NPU固件       | 24.1.RC1 |
  |      昇腾NPU驱动       | 24.1.RC1 |

## 安装模型环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 2**  版本支持表

  |      三方库      |   支持版本   |
  |:--------------:|:--------:|
  |    PyTorch     | 2.1，1.11 |
  | mmsegmentation |  1.2.2   |
  |      mmcv      |  2.1.0   |
  |    mmengine    |  0.10.3  |

- 安装依赖

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  
  ```
  pip install -r opencv_version.txt
  
  # pyTorch 1.11请使用requirements_1_11.txt，pytorch 2.1请使用requirements_2_1.txt
  pip install -r requirements_1_11.txt
  
  # 依照编译mmcv章节进行mmcv安装
  ```

- 编译mmcv 2.1.0

  在官网获取mmcv 2.1.0版本，解压至mmcv文件夹，运行以下命令
  ```
  cd mmcv
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```
# 准备数据集

## 预训练数据集
用户自行获取*Cityscapes*数据集，将数据解压到工程的data目录下，执行以下脚本进行数据集的预处理
```
# --nproc表示8个进程进行转换，也可以省略
python tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```

## 获取预训练权重
1. 联网情况下，预训练权重会自动下载。
2. 无网络情况下，用户可以访问pytorch官网自行下载*resnet18*预训练*resnet18-f37072fd.pth*。获取对应的预训练模型后，将预训练文件拷贝至对应目录。
```
${torch_hub}/checkpoints/resnet18-f37072fd.pth
```

# 快速开始

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
     bash ./test/train_1p.sh --data_path=指向cityscapes数据集 # 单卡精度
     
     bash ./test/train_1p.sh --data_path=指向cityscapes数据集 --performance=1  # 单卡性能
     ```
   
- 单机8卡训练

     启动8卡训练。
   
     ```
     bash ./test/train_8p.sh --data_path=指向cityscapes数据集  # 8卡精度

     bash ./test/train_8p.sh --data_path=指向cityscapes数据集 --performance=1 # 8卡性能 
     ```

   --data_path参数填写数据集路径，需写到cityscapes的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                              //数据集路径
   --performance                            //--performance=1开启性能测试，默认不开启
   --work_dir                               //输出路径包括日志和训练参数
   --resume                                 //--resume=1开启断点续训，默认不开启
   --batch_size                             //单卡batch_size，默认为2
   --num_workers                            //dataloader的workers数量，默认为2
   ```

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


## 训练结果

**表 3**  训练结果展示表

|  芯片      | 卡数 | mIoU  |  FPS   | Max Iters |
|:--------:|----|:-----:|:------:|:---------:|
|   GPU    | 1p |   -   | 19.98  |   1000    |
|   GPU    | 8p | 70.73 | 131.63 |   80000   |
| Atlas A2 | 1p |   -   | 16.25  |   1000    |
| Atlas A2 | 8p | 71.60 | 93.21  |   80000   |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明
2024.03.08：首次发布。

## FAQ
暂无。



