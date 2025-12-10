# SigLIP2(TorchAir)-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******

# 概述
SigLIP 是改进损失函数后的 CLIP 多模态模型，可用于零样本图像分类、图文检索等任务。
SigLIP 2模型在SigLIP的原始训练目标基础上，增加了额外的目标，以提升语义理解、定位和密集特征提取能力。SigLIP 2兼容SigLIP，用户只需要更换模型权重和tokenizer，可以按照相同的步骤进行模型推理。
本文档以 siglip2-so400m-patch14-384 为例介绍TorchAir在线推理步骤。


# 推理环境准备
- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 25.2.0 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                               | 8.2.RC1 | -                                                            |
| Python                                                       | 	3.11  | -                                                            |
| PyTorch                                                      | 2.4.0 | -                                                            |
| Ascend Extension PyTorch                                                     | 2.4.0.post2  | -                                                            |
| 说明：Atlas 800T A2 推理卡请以CANN版本选择实际固件与驱动版本 | \       | \                                                            |


# 快速上手

## 获取源码


1. 获取本仓源码。

   ```shell
   git clone https://gitcode.com/Ascend/ModelZoo-PyTorch.git  
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/foundation_models/SigLIP2/TorchAir
   ```


2. 安装推理所需依赖。

   ```shell
   pip3 install -U pip && pip3 install -r ../requirements.txt
   ```

## 模型推理
1. 下载模型权重。

      ```
      mkdir models
      ```

      下载[模型权重](https://huggingface.co/google/siglip2-so400m-patch14-384)siglip2-so400m-patch14-384置于models目录下。

2. 获取transformers源码。
 
    ```shell
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    git checkout v4.51.0
    ```

3. 在transformers源码目录下执行patch文件。
    ```shell
    git apply ../adapt_torchair.patch
    ```

4. 下载[样例图片](http://images.cocodataset.org/val2017/000000039769.jpg)命名为zero_shot_test_image.jpg置于TorchAir目录下。

5. 在TorchAir目录下设置环境变量，再执行推理命令。

    ```shell
    export PYTHONPATH=transformers/src/:$PYTHONPATH
    python3 torchair_infer.py \
       --pytorch_ckpt_path models/siglip2-so400m-patch14-384 \
       --image_path zero_shot_test_image.jpg \
       --candidate_labels "2 cats, a plane, a remote" \
       --batch_size 16 \
       --loop 10 \
       --device_id 0 
    ```
    - 参数说明
      -  --pytorch_ckpt_path：模型权重路径。
      -  --image_path：输入图片，默认为zero_shot_test_image.jpg。
      -  --candidate_labels：候选标签，默认为"2 cats, a plane, a remote"。
      -  --batch_size： 默认为16。
      -  --loop：性能测试的循环次数，默认为10。
      -  --device_id：npu芯片id，默认为0。
  
    推理脚本以计算文本、图像的特征为例，推理后将打屏模型性能结果。
6. 精度验证。
    本模型使用[ImageNet ILSVRC 2012](https://www.image-net.org/challenges/LSVRC/index.php)验证集进行推理测试。 
    1. 获取原始数据集置于TorchAir目录下。（解压命令参考tar -xvf  \*.tar与 unzip \*.zip）
      用户自行获取数据集后，上传数据集并解压。数据集目录结构如下所示：

    ```
    ImageNet/
    |-- val
    |   |-- ILSVRC2012_val_00000001.jpeg
    |   |-- ILSVRC2012_val_00000002.jpeg
    |   |-- ILSVRC2012_val_00000003.jpeg
    |   ...
    |-- val_label.txt
    ...
    ```
    2. 下载[标签集合文件](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)命名为imagenet1000_clsidx_to_labels.txt置于TorchAir目录下。
    3. 执行精度测试脚本。
      推理结果与数据集真值标签val_label.txt比对，打屏Accuracy数据。
      ```shell
      python3 torchair_test_accuracy.py --val_dir ImageNet/val \
      --gt_file ImageNet/val_label.txt \
      --pytorch_ckpt_path models/siglip2-so400m-patch14-384 \
      --classnames_file imagenet1000_clsidx_to_labels.txt \
      --batch_size 32 \
      --device_id 0
      ```

      - 参数说明
        -  --val_dir：数据集所在路径。
        -  --gt_file：真值标签文件所在路径。
        -  --pytorch_ckpt_path：模型权重路径。
        -  --classnames_file：标签集合文件。
        -  --batch_size： 默认为32。
        -  --device：npu芯片id，默认为0。

# 模型推理性能&精度
模型的纯推理性能参考如下数据。
- 提取文本特征
| 硬件形态     | Input Shape(labelnums x seqlen) | Texts Per Second |
|:-----------|:------------|:------------:|
| Atlas 800T A2 (8*64G) 单卡    | 1 x 64  |  122.92    |
| Atlas 800T A2 (8*64G) 单卡    | 3 x 64  |  65.77    |
| Atlas 800T A2 (8*64G) 单卡    | 10 x 64  |  43.53  |
| Atlas 800T A2 (8*64G) 单卡    | 100 x 64  |  7.74    |
| Atlas 800T A2 (8*64G) 单卡    | 1000 x 64  |  0.72    |

- 提取图像特征 
| 硬件形态    | Input Shape(bs x channels x height x width) | Frames Per Second |
|:-----------|:------------|:------------:|
| Atlas 800T A2 (8*64G) 单卡  | 1 x 3 x 384 x 384        |  40.49   |
| Atlas 800T A2 (8*64G) 单卡  | 4 x 3 x 384 x 384        |  57.34    |
| Atlas 800T A2 (8*64G) 单卡  | 8 x 3 x 384 x 384        |  58.51   |
| Atlas 800T A2 (8*64G) 单卡  | 16 x 3 x 384 x 384        |  61.04    |
| Atlas 800T A2 (8*64G) 单卡  | 24 x 3 x 384 x 384        |  60.63    |
| Atlas 800T A2 (8*64G) 单卡  | 32 x 3 x 384 x 384        |  59.58   |

模型的推理精度参考如下数据。
| 硬件形态 |  数据集|   精度 |    参考精度(GPU A10)|
|:------:|:------:|:----------:|:---------:|
|   Atlas 800T A2 (8*64G) 单卡 | ImageNet| top1: 74.12|top1: 74.10|