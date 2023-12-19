# DeepLabV3模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DeeplabV3是一个经典的图像语义分割网络，在v1和v2版本基础上进行改进，多尺度(multiple scales)分割物体，设计了串行和并行的带孔卷积模块，采用多种不同的atrous rates来获取多尺度的内容信息，提出 Atrous Spatial Pyramid Pooling(ASPP)模块, 挖掘不同尺度的卷积特征，以及编码了全局内容信息的图像层特征，提升图像分割效果。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  branch=master
  commit_id=fa1554f1aaea9a2c58249b06e1ea48420091464d
  model_name=DeeplabV3
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 520 x 520 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 21 x 520 x 520 | NCHW           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套  | 版本  | 环境准备指导  |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>


1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持coco2017数据集。用户需自行获取数据集，其目录结构如下：

   ```
   coco2017
   ├── val2017                 //验证集图片信息       
   └── annotations             // 验证集标注信息
   ```
  


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.ts文件。

   1. 导出ts模型。

      1. 使用`export.py`导出ts文件（请确保网络可连接至torch.hub）。

         ```
         python3 export.py
         ```

         获得`deeplabv3_resnet50.ts`文件。

   2. 改图

      1. 将刚获得的ts文件解压，`unzip deeplabv3_resnet50.ts`，获得`deeplabv3_resnet50`文件夹
      2. 将`deeplabv3_resnet50/code/__torch__/torchvision/models/segmentation/deeplabv3.py`替换为本项目提供的`deeplabv3.py`
      3. 完成替换后，重新压缩：`zip -r deeplabv3_resnet50.zip deeplabv3_resnet50`

   3. 精度测试

      1. 使用`run.py`执行数据集上的模型推理

         ```
         python3 run.py --dataset_path ./coco2017 --ts_path ./deeplabv3_resnet50.zip
         ```

        - 参数说明
          - dataset_path：数据集所在目录
          - ts_model：模型文件路径

   4. 性能验证

      1. 使用`perf.py`执行PSENet的性能测试

         ```
         python3 perf.py --mode ts --ts_path ./deeplabv3_resnet50.zip --batch_size 1 --opt_level 1
         ```

        - 参数说明
          - mode：使用ts模型进行推理
          - ts_path：ts模型文件所在路径
          - batch_size：batch数
          - opt_level：模型优化参数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|     310P3     |     1       |   coco2017     |   mIOU=64.5%   |   66FPS   |
