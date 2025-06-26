# BERT-NER-CRF for PyTorch
- [概述](#概述) 
- [准备训练环境](#准备训练环境)
- [准备数据集](#准备数据集)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

## 概述
### 简述
BERT-CRF 是用于自然语言处理中实体识别任务的模型
* 参考实现 https://github.com/lonePatient/BERT-NER-Pytorch
* 本代码仓为适配NPU的实现

## 准备训练环境

该模型为随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），您可以根据下面提供的安装指导选择匹配的CANN等软件下载使用。

### 准备环境

- 推荐使用最新的版本准备训练环境。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 25.0.RC1.1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.0.RC1.1 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.1.RC1 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v7.0.0-pytorch2.1.0 </td>
    </tr>
  </table>

- 三方库依赖如下表所示
  
  **表 2** 三方库依赖表

  | Torch_Version      | 三方库依赖版本                      |
  | :--------: | :----------------------------------------: |
  | PyTorch 2.1   | transformers 4.29.2 |

- 安装依赖

  在模型根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```

### 准备数据集
* 本模型在[Cluener](https://www.cluebenchmarks.com/introduce.html)数据集上完成训练和验证。在https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip 下载Cluener数据集，解压后放到datasets目录下，形成如下的目录结构：

```
BERT-NER-Pytorch
└── datasets
    ├── cner
    └── cluener 
        ├── cluener_predict.json
        ├── dev.json
        ├── __init__.py 
        ├── README.md
        ├── test.json
        └── train.json
```

### 准备预训练权重
* 在https://huggingface.co/bert-base-chinese/tree/main/ 下载预训练权重和config文件等相关信息，放在prev_trained_model目录下，形成如下的目录结构：

```
BERT-NER-Pytorch
└── prev_trained_model
    └── bert-base-chinese
        ├── config.json
        ├── pytorch_model.bin 
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── vocab.txt
```

## 开始训练
### 运行训练脚本

- 单机8卡训练

    ```
    bash test/train_full_8p.sh      # 8卡精度训练
    bash test/train_performance_8p.sh    # 8卡性能训练  
    ```

- 单机16卡训练

    ```
    bash test/train_full_8p.sh     # 16卡精度训练
    bash test/train_performance_16p.sh    # 16卡性能训练  
    ```


训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

## 训练结果展示
**表 3** 

| Name | F1 | Wps     | Samples/Second | Epochs |
| --- | --- |---------| --- | --- |
| 8p-NPU | 79.16 | 1163.21 | 1129.4 | 4 | 

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| 8p-竞品 | FP32 | 1942.15 |
| 8p-Atlas 900 A2 PoDc | FP32 | 1407.21 |

## 版本说明
### 变更
2023.6.19 首次发布

### FAQ
1. 若遇到safetensors三方库报这个错误“safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge”，原因是accelerate版本 >= v0.25.0，会默认使用safetensors，导致报错。参考解决方法，安装0.24.1版本的accelerate。
   ```
   pip install accelerate==0.24.1
   ```
