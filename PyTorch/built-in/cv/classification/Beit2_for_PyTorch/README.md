# BEiT v2 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述 
BEiT v2是基于矢量量化视觉标记的掩码图像模型


- 参考实现：

  ```
  url=https://github.com/microsoft/unilm/tree/master/beit2
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/classification/Beit2_for_PyTorch
  ```

拉取模型代码到本地并下载bpe_simple_vocab_16e6.txt.gz：

```
    git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git

    cd PyTorch/built-in/cv/classification/Beit2_for_PyTorch/vqkd_teacher/clip/
    wget https://github.com/microsoft/unilm/blob/master/beit2/vqkd_teacher/clip/bpe_simple_vocab_16e6.txt.gz

    cd ../..
    mkdir result
```

# 准备训练环境

## 准备环境

默认配置需要每张卡有60G空闲内存。

该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**

- 当前模型支持的 PyTorch 历史版本和已知三方库依赖如下表所示。

  **表 1**  环境配置表

| Software  |  Version |  Link |
|---|---|---|
| Python    | 3.8.18   |——|
| driver  |  23.0.RC3.1 | [link ](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/261425590?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743)  |
| firmware|  6.4.0.5.220 | [link ](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/261425590?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743)  |
|  CANN  |  Ascend-cann-toolkit_7.0.0.alpha002_linux | [link](https://ascend-repo.obs.myhuaweicloud.com/CANN_Test/20231116/Ascend-cann-toolkit_7.0.0.alpha002_linux-x86_64.run) [link](https://ascend-repo.obs.myhuaweicloud.com/CANN_Test/20231116/Ascend-cann-toolkit_7.0.0.alpha002_linux-aarch64.run)  |
| binary arithmetic package  | Ascend-cann-kernels-910b_7.0.0.alpha002_linux  | [link](https://ascend-repo.obs.myhuaweicloud.com/CANN_Test/20231116/Ascend-cann-kernels-910b_7.0.0.alpha002_linux.run)  |
| PyTorch 2.1 | torchvision==0.16.0 |
| PyTorch 2.2 | torchvision==0.17.0 |
| PyTorch 2.3 | torchvision==0.18.1 |
| PyTorch 2.4 | torchvision==0.19.0 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
   
  
- 安装流程。
在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
```
    # python3.8
    conda create -n beit2env python=3.8
    conda activate beit2env

    pip install -r 2.1_requirements.txt  # PyTorch2.1版本
    pip install -r 2.2_requirements.txt  # PyTorch2.2版本
    pip install -r 2.3_requirements.txt  # PyTorch2.3版本
    pip install -r 2.4_requirements.txt  # PyTorch2.4版本
```

## 准备数据集
```
    mkdir /dataset
```
   数据集下载路径：https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset?resource=download-directory
   
   下载并解压后将coco2017文件夹放到/dataset 下

   目录结构参考如下：

   ```
   /dataset/coco2017
  └── train2017
      ├── 000000000009.jpg
      ├── 000000000025.jpg
      ├── 000000000030.jpg
          ...
   ```
## 准备权重文件
```
mkdir tokenizer_model
cd tokenizer_model
wget https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D
cd ..
```

# 开始训练

## 训练模型

- 修改pretraining.sh

   ```
    # 修改此处的xxxx为CANN包安装路径
    source  /xxxx/ascend-toolkit/set_env.sh
    # 修改数据集路径
    data_path=./dataset/coco2017
    output_dir=./result
    log_dir=./result

   ```
   
   其他参数说明如下。
   

   ```
    公共参数：
    --epochs                              //训练总迭代数
    --batch_size                          //每个epoch中iteration个数
    --model                               //待训练模型名称
    --log_dir                             // log 存放路径
    --output_dir                          // path where to save, empty for no saving
    --data_path                           //数据集路径
    --layer_scale_init_value              //0.1 for base, 1e-5 for large. set 0 to disable layer scale
    --num_mask_patches                    //number of the visual tokens/patches need be masked
    --input_size                          //images input size for backbone
    --second_input_size                   //images input size for discrete vae
    --drop_path                           //Drop path rate (default: 0.1)

   ```


- 运行训练脚本

   该模型预训练支持单机8卡训练。

   ```
   bash ./pretraining.sh
   ```

## 训练结果展示

**表 2**  训练结果展示表

这里使用了单机进行预训练，采用的NPU型号为Atlas 900 A2 PODc

| NAME  | single-step time  | Iterations  | DataType|Torch_Version |
|:-:|:-:|:-:|:-:|:-:|
| NPU  | 4.417| 5000   | BF16|2.1.0|
| Reference| 4.316  | 5000   |BF16|2.1.0  |

精度loss5000步：
![](raw/beit2_5b_bf16_loss_compare.png)

# 版本说明

## 变更

2023.9.30：首次发布。

2023.11.15：性能优化。
