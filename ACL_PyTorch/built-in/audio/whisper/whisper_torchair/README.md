# Whisper(TorchAir)-推理指导

## 概述
Whisper 是 OpenAI 开源的通用语音识别模型，支持多语言转录和翻译，基于 Transformer 架构，适用于会议记录、字幕生成等场景。使用 torchair 编译模型加速。该推理指导使用了whisperx的推理流程，结合了 funasr 的 VAD 模型进行语音切分，以及 transformer pipeline 组batch功能。支持whisper-base, whisper-large-v3以及whisper-large-v3-turbo。

## 插件与驱动准备

- 该模型需要以下插件与驱动

  | 配套                                                            | 版本          | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    |-------------| ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 25.0.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 8.2.RC1     | 包含kernels包和toolkit包                                                                                                   |
  | Python                                                          | 3.11        | -                                                                                                     |
  | PyTorch                                                         | 2.5.1       | -                                                                                                     |
  | Ascend Extension PyTorch                                        | 2.5.1 | -                                                                                                     |
  | 说明：支持Atlas 800I A2/Atlas 300I A2，不支持Atlas 300I DUO/Atlas 300I Pro | \           | \                                                                                                     |


## 获取本仓源码
```
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/whisper/whisper_torchair
```

## 环境准备

* 通过以下命令下载并安装（或升级至）Whisper 的最新版本：

  `pip3 install -U openai-whisper`

* 下载模型权重：
```
mkdir weight
cd weight
```
whiper .pt格式权重：
  * `base.pt`: [下载链接](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)
  * `large-v3.pt`：[下载链接](https://modelscope.cn/models/iic/Whisper-large-v3/files)
  * `large-v3-turbo.pt`：[下载链接](https://modelscope.cn/models/iic/Whisper-large-v3-turbo/files)

whisper .safetensors权重：
  * whisper-base safetensors: [下载链接](https://modelscope.cn/models/openai-mirror/whisper-base/files)
  * whisper-large-v3 safetensors：[下载链接](https://modelscope.cn/models/openai-mirror/whisper-large-v3/files)
  * whisper-large-v3-turbo safetensors：[下载链接](https://modelscope.cn/models/openai-mirror/whisper-large-v3-turbo/files)

VAD权重：
  * speech_fsmn_vad_zh-cn-16k-common-pytorch: [下载链接](https://huggingface.co/alextomcat/speech_fsmn_vad_zh-cn-16k-common-pytorch/tree/main)
```
cd ..
```

* 权重转换（safetensor 转换成 pt 格式）：

  如果下载的是 pt 格式的权重可以忽略这一步

  `python3 weight_converter.py --model_name large-v3 --model_path ./weight/whisper-large-v3 # model_name有效参数为 large-v3, large-v3-turbo 和 base，model_path按具体情况修改`

* 安装命令行工具**ffmpeg**：
  * 在 Ubuntu or Debian上:
    `sudo apt update && sudo apt install ffmpeg`
  * 在 Arch Linux上:
    `sudo pacman -S ffmpeg`

* 安装requirements：
  `pip3 install -r requirements.txt`

## 数据集准备
* LibriSpeech/dev-clean数据集[下载地址](https://www.openslr.org/12)
* `audio.mp3`是普通的语音文件，可以直观测试，可以通过以下链接获取。（你也可以自己找一个中文语音.mp3/wav文件，放入目录中）
  ```TEXT
  https://pan.baidu.com/s/1Yvln3t88XbOR5bfDPdLByg?pwd=i3x8 提取码: i3x8 复制这段内容后打开百度网盘手机App，操作更方便哦
  ```

## 文件目录结构
文件目录结构大致如下：

```text
📁 whisper_torchair/
├── check_numa.sh
├── audio.mp3
├── infer.py
├── modeling_whisper.py
├── pipeline.py
├── test_performance.py
├── transcribe.py
├── 📁 LibriSpeech/
├── 📁 patches/
|   └── 📄 patch_apply.py
|   └── 📄 kaldi.patch
|   └── 📄 vad_model.patch
|   └── 📄 wav_frontend.patch
├── README.md
├── requrements.txt
├── run_wer_test.py
├── weight_converter.py
├── 📁 weight/
|   |── 📁 Whisper-large-v3
│       └── 📄 large-v3.pt
|   |── 📁 speech_fsmn_vad_zh-cn-16k-common-pytorch
```

## 模型推理
**脚本功能说明**：
- infer.py主要用于短音频（<30s）的转录以及LibriSpeech数据集的性能验证
- transcribe.py用于长音频转录，如智慧教室生成字幕场景

1. 激活环境变量
    ```SHELL
    source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 具体路径根据你自己的情况修改
    # 提升性能相关环境变量
    export TASK_QUEUE_ENABLE=1
    export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
    ```

2. 指定使用NPU ID，默认为0
    ```SHELL
    export ASCEND_RT_VISIBLE_DEVICES=0
    ```
3. 给funasr和torchaudio打补丁
    ```SHELL
    cd patches
    python3 patch_apply.py
    cd ..
    ```
4. 使能绑核，进一步提升性能
    ```SHELL
    export CPU_AFFINITY_CONF=1
    apt-get update
    apt-get install numactl
    # 在容器外执行脚本查看NPU id对应的NUMA node和cpu
    bash check_numa.sh
    ```
    回显如下：
    ```SHELL
    ...
    >>>>设备 0 对应 NUMA 节点: 6, NUMA node6 CPU(s):     192-223
    ...
    ```
5. 短音频推理demo, 根据实际查询到的核数配置，比如
    ```SHELL
    taskset -c 192-223 python3 infer.py --whisper_model_path ./weight/Whisper-large-v3/large-v3.pt
    ```
infer.py推理参数：
* --whisper_model_path：whisper模型权重路径，默认为"./weight/Whisper-large-v3/large-v3.pt"
* --audio_path：音频文件的路径，默认为"audio.mp3"
* --batch_size: batch_size大小，默认为1
* --warmup：warm up次数，默认为3，首次warm up时编译成图

6. 性能测试，使用LibriSpeech数据集，采用whisperx推理流程，先通过vad切分音频再重组batch。
    ```SHELL
    taskset -c 192-223 python3 test_performance.py --whisper_model_path ./weight/Whisper-large-v3/large-v3.pt --vad_model_path ./weight/speech_fsmn_vad_zh-cn-16k-common-pytorch 
    ```
* --whisper_model_path：whisper模型权重路径，默认为"./weight/Whisper-large-v3/large-v3.pt"
* --vad_model_path：vad模型权重路径，默认为"./weight/speech_fsmn_vad_zh-cn-16k-common-pytorch"
* --audio_path：音频文件的路径，默认为"audio.mp3"
* --librispeech_perf_test：启用该参数时，将在LibriSpeech数据集的部分数据上进行性能测试，并输出结果及转录比。默认值为 True
* --skip_librispeech_perf_test：传入该参数时，跳过LibriSpeech数据集的性能测试
* --speech_path：librispeech dev clean数据集文件的路径，默认为"./LibriSpeech/dev-clean/"
* --num_audio_files：从librispeech dev clean数据集中选取部分音频文件做性能测试，默认为52个，调整音频数量尽量让vad切分合并后的segment数接近但不大于batch size来达到最高性能
* --librispeech_wer_demo：启用该参数时，将对一条LibriSpeech音频数据做转录，并计算wer。默认值为 True
* --skip_librispeech_wer_demo：启用该参数时，跳过wer精度demo
* --device: npu设备编号，默认为0
* --batch_size: batch_size大小，默认为16
* --warmup：warm up次数，默认为4，首次warm up时编译成图

6. 长音频转录：
    ```SHELL
    taskset -c 192-223 python3 transcribe.py --whisper_model_path ./weight/Whisper-large-v3/large-v3.pt --audio_path {audio_file}
    ```
transcribe.py参数说明：
* --whisper_model_path：whisper模型权重路径，默认为"./weight/Whisper-large-v3/large-v3.pt"
* --language：输出语言，默认为中文
* --sample_audio：warm up阶段使用的音频，默认为"audio.mp3"
* --audio_path：长音频文件路径，必选参数
* --device: npu设备编号，默认为0
* --warmup：warm up次数，默认为3，首次warm up时编译成图

## 性能数据
  infer.py取librispeech dev clean数据集中的部分音频进行转录，性能如下

  | 模型                     | 芯片 | 平均转录比 |
  |-------------------------|------|----------|
  |whisper-base     |  800I A2 32G| 400       |
  |whisper-large-v3      |  800I A2 32G| 70       |
  |whisper-large-v3-turbo|  800I A2 32G| 170      |

## 精度测试
  执行以下命令来对librispeech dev clean数据集做全量的精度测试
  ```
  python3 run_wer_test.py --whisper_model_path ./weight/Whisper-large-v3/ 
  ```
  | 模型                     | 芯片 | WER         | 竞品WER|
  |-------------------------|------|-------------| ------ |
  |whisper-base      |  800I A2 32G| 0.085  | 0.086|
  |whisper-large-v3      |  800I A2 32G| 0.049  | 0.051 |
  |whisper-large-v3-turbo|  800I A2 32G| 0.050  | 0.051 |