# WhisperX推理指导

- [WhisperX推理指导](#whisperx推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型编译](#模型编译)
  - [模型推理](#模型推理)

# 概述

该工程使用mindietorch部署WhisperX模型


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套   | 版本        | 环境准备指导 |
  |-----------| ------- | ------------ |
  | Python | 3.10.13   | -            |
  | torch  | 2.1.0+cpu | -            |
  | torch_audio  | 2.1.0+cpu | -            |
  | CANN   | 8.0.B023  | -            |
  | MindIE | 1.0.B030  | -       |

# 快速上手
## 获取源码

1. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```


2. Whisper large V3模型权重下载路径:
    ```bash
    https://huggingface.co/openai/whisper-large-v3/tree/main
    ```
    将权重文件存放至当前目录下的model_path文件夹，请先创建改文件夹。
    

3. WhisperX中VAD模型权重下载路径:
    ```bash
    https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin
    ```
    并修改文件名为`whisperx-vad-segmentation.bin`


4. 安装依赖
    ```
    pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    pip3 install nltk
    pip3 install librosa
    pip3 install transformers==4.36.0
    pip3 install numpy==1.24.0
    pip3 install ml-dtypes
    pip3 install cloudpickle
    pip3 install pyannote.audio==3.1.1
    ```
    同时需要保证环境安装了libsndfile1, ffmpeg库

## 模型编译
1.1 300IPro环境执行如下命令：
```
python3 compile_300IPro_whisper.py \
-model_path ./model_path \
-soc_version Ascend310P3 \
```
    参数说明：
      - -model_path：预训练模型路径,必选。
      - -bs：batch_size, 默认值为32， 可选。
      - -save_path: 编译好的模型的保存文件，可选。默认值为"./compiled_models"。
      - -device_id: 选在模型运行的卡编号，默认值0，可选。
      - -soc_version: 芯片类型,必选。
    约束说明：
        1. 当前暂不支持动态batch，batch_size改变后，需要重新编图。

1.2 800IA2环境执行如下命令
```
python3 compile_800IA2_whisper.py \
-model_path ./model_path \
-soc_version Ascend910B4
```
    参数说明：
      - -model_path：预训练模型路径,必选。
      - -bs：batch_size, 默认值为16， 可选。
      - -save_path: 编译好的模型的保存文件，可选。默认值为"./compiled_models"。
      - -device_id: 选在模型运行的卡编号，默认值0，可选。
      - -soc_version: 芯片类型,必选。
    约束说明：
        1. 当前暂不支持动态batch，batch_size改变后，需要重新编图。

2.0 应用VAD模型补丁
在编译VAD模型前需要先打补丁，使用如下命令
```
python3 patch_apply.py
python3 remove_script.py
```

2. VAD模型编译
    ```
    python3 compile_vad.py \
    -vad_model_path /vad_model_path \
    -soc_version soc_version
    ```

    参数说明：
      - -vad_model_path：VAD预训练模型路径,必选。
      - -save_path: 编译好的模型的保存文件，可选，默认值"./compiled_models"。
      - -device_id: 选在模型运行的卡编号，默认值0，可选。
      - -soc_version: 芯片类型,必选。

    注：VAD模型编译的保存路径需要和Whisper-large-V3模型编译保存路径一致


## 模型推理
1. 设置mindie内存池上限为32，执行如下命令设置环境变量。内存池设置过小，内存重复申请和释放会影响性能。
    ```
    export TORCH_AIE_NPU_CACHE_MAX_SIZE=32
    ```

2. 模型推理
   ```
    python3 pipeline.py \
    -whisper_model_path /whisper_model_path \
    -vad_model_path /vad_model_path \
    -machine_type machine_type \
    -audio_path /audio_path
    ```

    参数说明：
      - -model_path：预训练模型路径,必选。
      - -bs：batch_size, 默认值为16， 可选。针对300IPro需要设置成32。
      - -save_path: 编译好的模型的保存文件，可选，默认值"./compiled_models"。
      - -device_id: 选在模型运行的卡编号，默认值0，可选。
      - -machine_type: 机器类型，必选。支持800IA2和300IPro


