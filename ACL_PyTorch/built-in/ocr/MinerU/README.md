# MinerU(TorchAir)-推理指导

- [MinerU(TorchAir)-推理指导](#MinerU(TorchAir)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [获取权重](#获取权重)
  - [获取数据集](#获取数据集)
  - [执行推理](#执行推理)
  - [精度测试](#精度测试)

******

# 概述
MinerU是由上海人工智能实验室OpenDataLab团队开发的开源文档解析工具，致力于解决大模型（LLM）训练和RAG（检索增强生成）应用中高质量结构化数据的提取难题。其核心价值在于将复杂文档（如PDF、网页、电子书）转换为机器可读的Markdown、JSON格式，同时保留原始文档的语义逻辑与多模态元素。

- 版本说明：
  
  ```
  url=https://github.com/opendatalab/MinerU.git
  commit_id=de41fa58590263e43b783fe224b6d07cae290a33
  model_name=MinerU
  ```

# 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1**  版本配套表
  
  | 配套                                                      | 版本          | 环境准备指导                                                                                        |
  | ------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                   | 25.2.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                    | 8.3.0       | -                                                                                             |
  | Python                                                  | 3.11        | -                                                                                             |
  | PyTorch                                                 | 2.6.0       | -                                                                                             |
  | Ascend Extension PyTorch                                | 2.6.0       | -                                                                                             |
  | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |

# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   
   ```
   git clone https://github.com/opendatalab/MinerU.git
   cd MinerU
   git reset --hard de41fa58590263e43b783fe224b6d07cae290a33
   pip3 install -e .
   cd ..
   ```

2. 安装依赖  
   
   ```
   pip3 install -r requirements.txt
   ### 此外，还需安装 Torchvision Adapter
   git clone https://gitee.com/ascend/vision.git vision_npu
   cd vision_npu
   git checkout v0.21.0-7.1.0
   pip3 install -r requirement.txt
   source /usr/local/Ascend/ascend-toolkit/set_env.sh # Default path, change it if needed.
   python setup.py bdist_wheel
   cd dist
   pip install torchvision_npu-0.21.0+git22ca6b2-cp311-cp311-linux_aarch64.whl
   cd ../../
   ```


3. 修改第三方库
进入第三方库安装路径，默认为`source_path = /usr/local/lib/python3.11/site-packages`,通过工作目录`workdir`(自定义)中的`ultralytics.patch`和`doclayout_yolo.patch`进行修改
   ```
   source_path=/usr/local/lib/python3.11/site-packages
   cd ${source_path}/ultralytics
   patch -p2 < ${workdir}/ultralytics.patch
   cd ${source_path}/doclayout_yolo
   patch -p2 < ${workdir}/doclayout_yolo.patch
   cd ${workdir}
   patch -p0 < mfr_encoder_mhsa.patch
   ```

## 获取权重

运行以下指令，下载权重文件[Model weights](https://www.modelscope.cn/models/OpenDataLab/PDF-Extract-Kit-1.0/summary)，默认存放为`/root/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0`

```
mineru-models-download --source modelscope --model_type pipeline
```
下载完成后，默认在根目录生成`mineru.json`文件，移动数据集时，需修改`/root/mineru.json`文件中"models-dir": "pipeline"为修改后权重存放路径

权重目录大致结构为：
```text
📁 models
├── 📁 Layout
│   └── 📁 YOLO
│       └── doclayout_yolo_docstructbench_imgsz1280_2501.pt
├── 📁 MFD
│   └── 📁 YOLO
│       └── yolo_v8_ft.pt
├── 📁 MFR
│   └── 📁 unimernet_hf_small_2503
│       ├── model.safetensors
│       ├── ……
│       └── tokenizer_config.json
├── 📁 OCR
│   └── 📁 paddleocr_torch
│       ├── Multilingual_PP-OCRv3_det_infer.pth
│       ├── arabic_PP-OCRv3_rec_infer.pth
│       ├── ……
│       ├── ……
│       └── th_PP-OCRv5_rec_infer.pth
├── 📁 ReadingOrder
│   └── 📁 layout_reader
│       ├── config.json
│       └── model.safetensors
└── 📁 TabRec
    └── 📁 SlanetPlus
        └── slanet-plus.onnx
```


## 获取数据集

创建数据集目录`OmniDocBench_dataset`，下载多样性文档解析评测集`OmniDocBench`数据集的[pdfs和标注](https://opendatalab.com/OpenDataLab/OmniDocBench)，解压并放置在`OmniDocBench_dataset`目录下
文件目录格式大致如下：
   ```
   📁 workdir
    ├── infer.py
    ├── ……
    ├── 📁 MinerU
    └── 📁 OmniDocBench_dataset
        ├── OmniDocBench.json
        └── 📁 pdfs
            └── ***.pdf
   ```

## 执行推理

运行推理脚本infer.py

```
python3 infer.py --data_path=OmniDocBench_dataset --model_source=local
```

- 参数说明
  - data_path: 数据集路径
  - model_source: 模型源类型，local表示使用本地文件，modelscope/huggingface表示在线拉取权重

推理执行完成后，解析结果存放于`OmniDocBench_dataset/output/`目录，结果除了输出主要的 markdown 文件外，还会生成多个辅助文件用于调试、质检和进一步处理。

## 精度测试

使用`OmniDocBench`数据集配套评测代码测试精度。

1. 推理结果整理

   将解析结果文件夹中的markdown文件整理放置于同一目录，本例将所有markdown文件存放于OmniDocBench_dataset目录下的results_md文件夹
   ```
   cp OmniDocBench_dataset/output/*/auto/*.md OmniDocBench_dataset/results_md/
   ```

2. 获取测评源码并构建环境
   
   ```
   git clone https://github.com/opendatalab/OmniDocBench.git
   cd OmniDocBench
   git reset --hard dc96d812d219960773399c02ae8f89e4706120d4
   conda create -n omnidocbench python=3.10
   conda activate omnidocbench
   pip install -r requirements.txt
   ```

3. 测评配置修改

   修改`OmniDocBench`测评代码中的config文件，具体来说，我们使用端到端测评配置，修改configs/end2end.yaml文件中的ground_truth的data_path为下载的OmniDocBench.json路径，修改prediction的data_path中提供整理的推理结果的文件夹路径，如下：
   ```
   # -----以下是需要修改的部分 -----
   dataset:
      dataset_name: end2end_dataset
      ground_truth:
      data_path: ../OmniDocBench_dataset/OmniDocBench.json
      prediction:
      data_path: ../OmniDocBench_dataset/results_md
   ```

4. 精度测量结果

   配置好config文件后，只需要将config文件作为参数传入，运行以下代码即可进行评测：
   ```
   python pdf_validation.py --config ./configs/end2end.yaml
   ```

   在`OmniDocBench`数据集上的精度为：
   |模型|芯片|overall_EN|overall_CH|
   |------|------|------|------|
   |MinerU|300I DUO|0.1588|0.2527|
   |MinerU|800I A2 64G|0.1580|0.2510|

