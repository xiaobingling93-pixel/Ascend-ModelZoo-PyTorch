# MuseTalk-推理指导

- [MuseTalk-推理指导](#musetalk-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)

******

# 概述
&emsp;&emsp;‌`MuseTalk` 是一个实时的音频驱动唇部同步模型。该模型能够根据输入的音频信号，自动调整数字人物的面部图像，使其唇形与音频内容高度同步。这样，观众就能看到数字人物口型与声音完美匹配的效果。MuseTalk 特别适用于256 x 256像素的面部区域，且支持中文、英文和日文等多种语言输入。

- 版本说明：
  ```
  url=https://github.com/TMElyralab/MuseTalk
  commit_id=
  model_name=MuseTalk
  ```

# 推理环境准备
- 该模型需要以下插件与驱动
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 25.0.RC1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  8.1.RC1 | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                          |  3.10 | -                                                                                                     |
  | PyTorch                                                         | 2.1.0 | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.1.0.post10 | -                                                                                                     |
  | 说明：Atlas 300I DUO 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码

1. 获取开源模型源码
   ```
   git clone https://github.com/TMElyralab/MuseTalk
   cd MuseTalk
   git reset --hard 058f7dd
   ```
2. 下载[ffmpeg-static](https://www.johnvansickle.com/ffmpeg/old-releases/), 并放置于path/to/ffmpeg文件夹下
   

3. 下载MuseTalk权重 [weights](https://huggingface.co/TMElyralab/MuseTalk).

4. 下载其他依赖权重（根据MuseTalk开源仓资料说明下载）:
   - [sd-vae-ft-mse，自动编码器，用于视频或图片特征的编解码](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [whisper，用于audio特征提取](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
   - [dwpose，用于人体姿态分析](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [face-parse-bisent，用于人脸解析](https://github.com/zllrunning/face-parsing.PyTorch)
   - [resnet18，用于人脸解析](https://download.pytorch.org/models/resnet18-5c106cde.pth)
   
   
   将下载后的权重放置于`models`文件夹下，本地下载完成后的目录树如下
   ```shell
    MuseTalk
    ├── models
    │   ├── musetalk
    │   │   └── musetalk.json
    │   │   └── pytorch_model.bin
    │   ├── dwpose
    │   │   └── dw-ll_ucoco_384.pth
    │   ├── face-parse-bisent
    │   │   ├── 79999_iter.pth
    │   │   └── resnet18-5c106cde.pth
    │   ├── sd-vae-ft-mse
    │   │   ├── config.json
    │   │   └── diffusion_pytorch_model.bin
    │   └── whisper
    │       └── tiny.pt
    ├── musetalk
    │   ├── models
    │   ├── utils
    │   └── whisper
    ├── scripts
    │   ├── inference.py
    │   ├── infer_npu.py      // 本仓库提供的自定义推理脚本
    |   ├── rewrite_models.py // 本仓库提供的文件
    │   ├── diff.patch        // 本仓库提供的patch文件
    │   ├── requirements.txt  // 本仓库提供的requirements文件
    │   ├── realtime_inference.py
    │   └── __init__.py
    ├── configs
    │   └── inference
    ├── data
    │   ├── audio
    │   └── video
    ├── app.py
    ├── entrypoint.sh
    ├── README.md
    ├── requirements.txt
    └── LICENSE
    ```


2. 安装依赖  
   ```
   pip3 install scripts/requirements.txt
   ```


## 模型推理

### 1 开始推理验证

   1. 设置环境变量，执行推理命令

      ```
      # 指定使用NPU ID，默认为0
      export ASCEND_RT_VISIBLE_DEVICES=0

      # 设置ffmpeg-static环境变量
      export FFMPEG_PATH=/path/to/ffmpeg

      # 应用patch文件
      git apply scripts/diff.patch

      # 执行推理命令
      python -m scripts.infer_npu --inference_config configs/inference/test.yaml (--use_float16)

      ```

      在推理开始后，首先会默认执行warm_up，目的是执行首次编译，首次编译时间较长，在warm_up结束后，会执行推理操作，并打屏计算结果和E2E性能数据。如果想测试模型推理耗时，可以在 `infer_npu.py` 文件 190行 `res_frame_list = infer(pd.input_latent_list_cycle, pd.whisper_chunks)` 前后添加时间打点。

### 2 性能

以MuseTalk仓内data/video/sun.mp4和data/audio/sun.wav为例，得到性能数据：

   |模型|芯片|E2E|forward|
   |------|------|------|---|
   |MuseTalk|Atlas 300I DUO|229s|26s|

   - forward包含vae decoder和unet模型的耗时
   - 将`config/inference/test.yaml`文件中的`task_0`注释掉
   - 开启如下环境变量可提升性能：
     - export ENABLE_TILING_CACHE=1
     - export TASK_QUEUE_ENABLE=2

### 3 FAQ
1. 首次推理会报依赖权重错误，将其下载后放到对应路径即可。
   ```
   Downloading: "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" to /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
   ```

