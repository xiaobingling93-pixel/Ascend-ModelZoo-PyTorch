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
### 准备环境
* 当前模型支持的PyTorch版本和已知三方库依赖如下表所示
表1 依赖库列表

| Torch_Version      | 三方库依赖版本                      |
| :--------: | :----------------------------------------: |
| PyTorch 1.11   | transformers 4.29.2 |
| PyTorch 2.1   | transformers 4.29.2 |

* 环境中也需要安装对应版本的CANN和torch_npu，可参考《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/softwareinstall/instg/instg_000018.html)》
* 安装依赖：
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
表2 

| Name | F1 | Wps     | Samples/Second | Epochs |
| --- | --- |---------| --- | --- |
| 8p-NPU | 79.16 | 1163.21 | 1129.4 | 4 | 

## 版本说明
### 变更
2023.6.19 首次发布

### FAQ
1. 若遇到safetensors三方库报这个错误“safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge”，原因是accelerate版本 >= v0.25.0，会默认使用safetensors，导致报错。参考解决方法，安装0.24.1版本的accelerate。
   ```
   pip install accelerate==0.24.1
   ```
