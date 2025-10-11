# Whisper模型(TorchAir)-推理指导

- [概述](#概述)
- [插件与驱动准备](#插件与驱动准备)
- [获取本仓源码](#获取本仓源码)
- [环境准备](#环境准备)
- [数据集准备](#数据集准备)
- [文件目录结构](#文件目录结构)
- [开始推理](#开始推理)
- [性能数据](#性能数据)

## 概述
Whisper 是 OpenAI 开源的通用语音识别模型，支持多语言转录和翻译，基于 Transformer 架构，适用于会议记录、字幕生成等场景。其特点是开箱即用、鲁棒性强，并提供多种模型尺寸平衡速度与精度。

## 插件与驱动准备

- 该模型需要以下插件与驱动

  | 配套                                                            | 版本          | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    |-------------| ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 25.0.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 8.1.RC1     | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                          | 3.10        | -                                                                                                     |
  | PyTorch                                                         | 2.5.1       | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.5.1 | -                                                                                                     |
  | 说明：Atlas 800I A2 推理卡和Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                                     |


## 获取本仓源码
```
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/whisper/
```

## 环境准备

* 通过以下命令下载并安装（或升级至）Whisper 的最新版本：

  `pip3 install -U openai-whisper`

* 下载模型权重：
  * `base.pt`：[下载链接](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)

* 安装命令行工具**ffmpeg**：
  * 在 Ubuntu or Debian上:
    `sudo apt update && sudo apt install ffmpeg`
  * 在 Arch Linux上:
    `sudo pacman -S ffmpeg`

* 安装requirements：
  `pip3 install -r requirements.txt`

## 数据集准备
* librispeech_asr_dummy数据集[下载地址](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/main)，该数据集是 Hugging Face Datasets 库中提供的一个小型测试数据集，用于快速验证语音识别。下载下来后，把它放入当前文件夹内。
* `audio.mp3`是普通的语音文件，在warm up阶段使用，并可以直观测试，可以通过以下链接获取。（你也可以自己找一个中文语音.mp3/wav文件，放入目录中）
  ```TEXT
  https://pan.baidu.com/s/1fHL0fWbGgKXQ9W1GXA2RBQ?pwd=xe2x 提取码: xe2x 复制这段内容后打开百度网盘手机App，操作更方便哦
  ```

## 文件目录结构
文件目录结构大致如下：
    
```text
📁 whisper/
├── audio.mp3
├── infer.py
├── rewrited_models.py
├── whisper_decoding.patch
├── base.pt
├── README.md
├── requrements.txt
├── 📁 librispeech_asr_dummy/
|   |── 📁 clean
│       └── 📄 validation-00000-of-00001.parquet
```

## 开始推理
```SHELL
# 1. 激活环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 具体路径根据你自己的情况修改
# 2. 指定使用NPU ID，默认为0
export ASCEND_RT_VISIBLE_DEVICES=0
# 3. 开始推理
python3 infer.py
```
infer.py推理参数：
* --model_path：模型权重路径，默认为"base.pt"
* --audio_path：音频文件的路径，默认为"audio.mp3"
* --speech_path：librispeech_asr_dummy数据集文件的路径，默认为"./librispeech_asr_dummy/clean/"
* --device: npu设备编号，默认为0
* --batch_size: batch_size大小，默认为1
* --warm_up：warm_up次数，默认为5
* --loop：循环测试次数，默认为5

在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，输出audio.mp3音频的推理得到的文本。

warmup结束之后，开始推理librispeech_asr_dummy数据集，推理过程中会打屏输出E2E性能，推理结束后会输出WER精度得分以及平均E2E时间。

## 性能、精度数据
  在librispeech_asr_dummy/clean数据集上的性能精度数据如下：

   | 模型      | 芯片     | 平均E2E时间 | WER |
   |---------|------------|----------|-------|
   | whisper base | 800I A2 64G | 71.73ms | 8.21% |
