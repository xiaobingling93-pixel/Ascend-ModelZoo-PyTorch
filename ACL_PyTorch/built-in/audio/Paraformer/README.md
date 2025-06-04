# Paraformer(TorchAir)-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

******

# 概述
Paraformer是阿里达摩院语音团队提出的一种高效的非自回归端到端语音识别框架。本项目为Paraformer中文通用语音识别模型，采用工业级数万小时的标注音频进行模型训练，保证了模型的通用识别效果。模型可以被应用于语音输入法、语音导航、智能会议纪要等场景。

- 版本说明：
  ```
  url=https://github.com/modelscope/FunASR
  commit_id=f47d43c
  model_name=paraformer
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            | 版本           | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    |--------------| ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 24.0.0       | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 8.0.0        | -                                                                                                   |
  | Python                                                          | 3.10         | -                                                                                                     |
  | PyTorch                                                         | 2.1.0        | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.1.0.post11 | -                                                                                                     |
  | 说明：Atlas 800I A2/Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \            | \                                                                                                     |


# 快速上手

## 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/Paraformer
   ```
   
1. 安装依赖  
   ```bash
   pip3 install -r requirements.txt
   ```

2. 获取模型仓源码
   ```bash
   git clone https://github.com/modelscope/FunASR.git
   cd FunASR
   git reset --hard 77db489a
   git apply ../adapt-torchair.patch
   pip3 install -e ./
   ```

3. 下载模型权重

    本文档以[Paraformer-large-热词版模型](https://www.modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404)为例进行说明。
    - 方式一
      
      手动下载权重文件并上传服务器。假定上传到当前 `FunASR` 同级目录。

    - 方式二

      通过git-lfs下载
      ```bash
      git lfs install
      git clone https://www.modelscope.cn/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404.git
      ```
4. 完整下载后的文件目录树如下

    ```shell
    Paraformer
    ├── FunASR      // 从开源代码仓下载的文件夹
    ├── speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404   // 模型权重下载
    ├── adapt-torchair.patch
    ├── infer.py           // 本仓库提供的自定义推理脚本
    ├── README.md
    └── requirements.txt
    ```

## 模型推理

1. 执行推理命令

    ```bash
    # 解决nan值导致的精度异常问题
    export INF_NAN_MODE_FORCE_DISABLE=1

    python3 infer_air.py \
       --model_path=./speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404 \
       --data=speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/example/asr_example.wav \
       --hotwords="魔搭" \
       --device=0 \
       --loop=10
    ```
    - 参数说明
      - model_path：模型权重路径
      - data：模型输入文件，默认为asr_example.wav
      - hotwords：语音热词，默认为“魔搭”
      - device：npu芯片id，默认为0
      - loop：性能测试的循环次数，默认为10
  
    推理脚本以计算单用例音频输出结果为例，推理后将打屏推理结果和模型性能

# 模型推理性能&精度
以800I A2为例

| 模型     | 硬件 | 端到端性能        |
|----------|------|-------------------|
| Paraformer |800I A2| 1.4 data/s |

