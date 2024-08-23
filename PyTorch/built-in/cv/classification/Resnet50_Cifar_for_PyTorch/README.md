# Resnet50-cifar for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [vNPU训练模型](vNPU训练模型.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

MMClassification 是一款基于 PyTorch 的开源图像分类工具箱，是 OpenMMLab 项目的成员之一。主要特性：支持多样的主干网络与预训练模型；支持配置多种训练技巧；大量的训练配置文件；高效率和高可扩展性；功能强大的工具箱。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmclassification
  commit_id=7b45eb10cdeeec14d01c656f100f3c6edde04ddd
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.11   | - |
  | PyTorch 2.1   | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```
  
- 安装mmcv。注意使用torch2.1版本时，-b分支需要指定为1.x。
  ```
  cd /${模型文件夹名称}
  git clone -b v1.7.0 --depth=1 https://github.com/open-mmlab/mmcv.git
  # pytorch版本大于等于2.1，需执行下列操作
  cp mmcv_need/setup.py mmcv/ 
  cp mmcv_need/text.py mmcv/mmcv/runner/hooks/logger/ 
  ```
  ```
  cd mmcv
  MMCV_WITH_OPS=1 pip3 install -e .
  
  # 备注：若mmcv编译较慢，建议安装ninja-build，加速编译安装。
  ```
  
- 安装mmcls。
  ```
  cd /${模型文件夹名称}
  pip3 install -e .
  ```


## 准备数据集

1. 获取数据集。

   模型训练所需要的数据集（cifar100），在训练过程中脚本会自动下载，请保持网络畅通。若下载失败，请用户自行下载该数据集。
   数据集目录结构参考如下所示。
   ```
   ├── cifar-100-python
      ├──file.txt   
      ├──train                  
      ├──meta     
      ├──test
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
     bash ./test/train_performance_1p.sh         # batchsize=16  单卡性能
     bash ./test/train_performance_1p_bs32.sh    # batchsize=32  单卡性能
     
     bash ./test/train_full_1p.sh                # batchsize=16  单卡精度
     bash ./test/train_full_1p_bs32.sh           # batchsize=32  单卡精度
     ```

   - 单机8卡训练

     启动8卡训练。
     ```
     bash ./test/train_performance_8p.sh         # batchsize=16  8卡性能
     bash ./test/train_performance_8p_bs32.sh    # batchsize=32  8卡性能
     
     bash ./test/train_full_8p.sh                # batchsize=16  8卡精度
     bash ./test/train_full_8p_bs32.sh           # batchsize=32  8卡精度
     ```

     注意：模型训练所需要的数据集（cifar100）脚本会自动下载，请保持网络畅通。如果已有数据集，也可用传参的方式传入，如以下命令：
     
     ```
     bash ./test/train_full_1p.sh --data_path=cifar100数据集路径
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --device                             //训练使用的设备
   --gpu-id                             //训练卡id指定
   --seed                               //随机数种子设置
   --world-size                         //分布式训练节点数
   --amp                                //设置是否使用混合精度训练
   --momentum                           //动量
   --weight-decay                       //权重衰减
   --batch-size                         //训练批次大小
   --lr                                 //初始学习率
   --epochs                             //重复周期数
   --data                               //数据集路径
   --print-freq                         //日志打印频率
   --addr                               //主机地址
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# vNPU训练模型

## 切分vNPU
- 执行以下命令设置虚拟化实例功能容器模式
   ```
   npu-smi set -t vnpu-mode -d 0
   ```
- npu-smi set -t create-vnpu -i id -c chip_id -f vnpu_config [-v vnpu_id] [-g vgroup_id] 用于创建指定模板的vNPU，也可指定vNPU的id和vNPU所属群组的id。

  vNPU内存不足会导致训练模型精度性能下降或无法拉起训练，切分模板选择vir12_3c_32g
  ```
  npu-smi set -t create-vnpu -i 0 -c 0 -f vir12_3c_32g -v 100
  ```
  
## 原生docker挂载vNPU
- 挂载vNPU，并声明shm内存（避免容器内存不足无法拉起训练）
   ```
   docker run -it \
  --device=/dev/vdavinci100:/dev/davinci100 \          # 挂载切分好的vNPU
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --shm-size=720g                                      # 指定shm-size
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /home:/home \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  docker_image_id  /bin/bash
   ```

- 初次启动容器，需要重新配置环境及相关依赖。

- 在搭载vNPU的容器内重新开始训练

# 训练结果展示

**表 2**  训练结果展示表

|  NAME  | Acc@1 |  FPS  | Epochs | AMP_Type | Torch_Version | batch_size | Device |
|:------:|:-----:|:-----:|:------:|:--------:|:-------------:|:----------:|:------:|
| 1p-NPU | -  | 4266  |   2    |    O2    |     1.11      |    256     | Atlas 800T A2  |
| 8p-NPU | 80.0  | 33032 |   2    |    O2    |     1.11      |    2048    | Atlas 800T A2  |
| 1p-NPU | -  | 4571  |   2    |    O2    |      2.1      |    256     | Atlas 800T A2  |
| 8p-NPU | 80.0  | 33032 |   2    |    O2    |      2.1      |    2048    | Atlas 800T A2  |

  > **说明：** 该模型默认在二进制场景下进行训练。

**表 3**  vNPU训练结果展示表

|    NAME     | Acc@1 |  FPS  | Epochs |   | Torch_Version | batch_size | Device  |
|:-----------:|:-----:|:-----:|:------:|:-:|:-------------:|:----------:|:-------:|
| 1p-NPU-ARM  | 88.517| 2007  |   90   |   |      2.1      |     16     |   Atlas 800T A2    |
| 1p-vNPU-ARM | 88.638 | 1372 |  90    |   |      2.1      |     16     |   Atlas 800T A2    |
| 1p-NPU-X86  |   88.322   |  1984  |   90   |   |      2.1      |     16     |   Atlas 200T A2 Box16      |
| 1p-vNPU-X86 |   88.450   |  969  |   90   |   |      2.1      |     16     |   Atlas 200T A2 Box16      |
同等超参下，vNPU能满足精度要求


# 版本说明

## 变更

2023.02.21：更新readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
