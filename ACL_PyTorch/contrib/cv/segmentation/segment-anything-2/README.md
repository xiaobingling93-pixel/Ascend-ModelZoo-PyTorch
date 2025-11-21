# segment-anything-2-real-time推理指导

## 概述

在实时视频流上运行 Segment Anything Model 2，适配Atlas 300I A2

- 参考实现：

  ```
  url=https://github.com/Gy920/segment-anything-2-real-time.git
  commit_id=e3623cda3f748df999f76db663f08da7a884a820
  model_name=segment-anything-2-real-time
  ```

### 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | frame    | FLOAT32  | batchsize x 3 x 512 x 512 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | masks    | FLOAT32  | num_objects x 1 x 256 x 256 | NCHW         |

## 推理环境准备

| 配套                                                         | 版本    | 环境准备指导 |
| ------------------------------------------------------------ | ------- | ------------ |
| 固件与驱动 | 25.2.3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies/pies_00001.html) |
| CANN                                                         | 8.2 | 包含kernels包和toolkit包 |
| Python                                                       | 3.11 | -            |
| PyTorch                                                      | 2.1.0 | -            |
| Ascend Extension Pytorch | 2.1.0 | - |
| 说明：Atlas 300I A2 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \            |

## 快速上手

### 获取源码

1. 获取本仓源码

   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/segmentation/segment-anything-2/
   ```

   

2. 在同级目录下，获取开源模型代码

   ```
   git clone https://github.com/Gy920/segment-anything-2-real-time.git
   cd segment-anything-2-real-time
   git reset --hard e3623cda3f748df999f76db663f08da7a884a820
   # 执行patch 适配昇腾推理卡
   git apply ../sam2.patch
   ```

3. 安装必要依赖。

   ```
   pip3 install -r ../requirements.txt
   ```

### 下载权重

```bash
cd checkpoints
./download_ckpts.sh
```

### 

### 准备数据集

参考[UItralytics](https://docs.ultralytics.com/zh/models/sam-2/#sa-v-dataset)使用[DAVIS 2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip)数据集用于精度测试，下载并上传到segment-anything-2-real-time的同级目录，解压为DAVIS，数据目录结构如下

```
segment-anything-2 
├── DAVIS
│   ├── Annotations_unsupervised
│   ├── ImageSets
│   ├── JPEGImages
│   ├── README.md
│   └── SOURCES.md
├── DAVIS-2017-Unsupervised-trainval-480p.zip
├── segment-anything-2-real-time
│   ├── assets
│   ├── checkpoints
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── demo
│   ├── fusion_result.json
│   ├── LICENSE
│   ├── LICENSE_cctorch
│   ├── notebooks
│   ├── pyproject.toml
│   ├── README.md
│   ├── result.gif
│   ├── sam2
│   └── setup.py
├── README
├── accuracy.py
├── sam.patch
├── requirements.txt

```

### 运行精度测试

```
cd ../..
mv acurracy.py segment-anything-2-real-time/demo/
cd segment-anything-2-real-time
python demo/acurracy.py
```

可通过如下参数指定精度测试的数据集路径，视频名称，模型权重，模型配置文件

| 参数名           | 默认值                                | 说明                                             |
| ---------------- | ------------------------------------- | ------------------------------------------------ |
| `--data_path`    | `../DAVIS`                            | 数据集路径，用于指定视频分割评估所需的数据集位置 |
| `--vdo_name`     | `bear`                                | 视频名称，指定要评估的具体视频文件或序列         |
| `--checkpoint`   | `./checkpoints/sam2.1_hiera_small.pt` | 模型检查点文件路径，用于加载训练好的模型权重     |
| `--model_config` | `configs/sam2.1/sam2.1_hiera_s.yaml`  | 模型配置文件路径，用于定义模型结构和参数         |

### 模型推理demo

```python
python demo/demo.py
```

## 模型推理性能与精度



| 芯片型号      | 精度(J&F) | 推理性能 |
| ------------- | --------- | -------- |
| Atlas 300I A2 | 0.9289    | 52ms     |