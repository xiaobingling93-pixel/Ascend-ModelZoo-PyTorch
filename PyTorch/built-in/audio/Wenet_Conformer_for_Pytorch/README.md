# Wenet Conformer for PyTorch

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持特性](#支持特性)
    - [代码实现](#代码实现)
-   [Wenet-Conformer](#Wenet-Conformer)
    - [准备环境](#准备环境)
    - [准备数据集](#准备数据集)
    - [开始训练](#开始训练)
    - [训练结果展示](#训练结果展示)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍
Wenet是一款开源的、面向工业落地应用的语音识别工具包，主要特点是小而精，它不仅采用了现阶段最先进的网络设计Conformer，还用到了U2结构实现流式与非流式框架的统一。

## 支持特性

本仓已支持以下模型任务类型。

| 模型        | 任务类型 | 是否支持  |
|-----------| ------- | ------------ |
| Conformer | 预训练 | ✅   |
| Whisper   | 预训练 | ✅   |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/wenet-e2e/wenet.git
  commit_id=ac9a2612e8245ac473a17f64eea600dd7afbeb20
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/audio
  ```

<a id="Wenet-Conformer"></a>
# Wenet-Conformer

## 准备环境

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

- 三方库依赖如下表所示。

  **表 2**  三方库依赖表

  | Torch_Version |       三方库依赖版本       |
  | ------------------- | ----------------- |
  | PyTorch 2.1   | torchaudio==2.1.0  |

- 安装依赖。

  在模型源码包根目录下执行命令。
  
  ```
  pip install -r requirements_2_1.txt
  ```

- 编译安装torchaudio

  在官网根据PyTorch版本获取torchaudio对应版本，解压至torchaudio文件夹，运行以下命令
  ```
  git clone https://github.com/pytorch/audio.git
  cd audio
  git checkout release/2.1
  pip install -r requirements.txt
  python setup.py develop
  ```

- 安装sox库
  ```bash
  #以yum为例，执行以下命令安装sox库及开发包
  sudo yum install sox sox-devel
  ```


## 准备数据集

1. 获取数据集。

   用户自行下载 `aishell-1` 数据集，并将下载好的数据集放置服务器的任意目录下。该数据集包含由 400 位说话人录制的超过 170 小时的语音。数据集目录结构参考如下所示。

   ```
    aishell-1
       ├── data_aishell.tgz
       |
       └── resource_aishell.tgz
   ```

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。

2. 确保权限设置正确。

  训练脚本在运行时会读取并生成诸如 `data/train/data.list` 等数据/中间文件。如果对数据集目录或模型目录没有执行权限，可能会出现如下错误：

  ```
  FileNotFoundError: [Errno 2] No such file or directory: 'data/train/data.list'
  ```

  该报错常见原因是目录或文件权限不足（例如从其他用户拷贝或解压得到的数据），而不是文件真实不存在。请在数据集目录和模型代码目录上为当前训练用户开启可执行权限，例如：

  ```bash
  chmod 755 -R /path/to/your/dataset
  chmod 755 -R /path/to/Wenet_Conformer_for_Pytorch
  ```

  > **说明：** 请根据实际部署环境调整路径，并确保仅在符合安全策略的前提下修改权限。


## 开始训练

### 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   Pytorch2.1版本，运行时可能出现段错误，如出现请参照[FAQ](#faq)对模型内部分相关代码进行修改。

   该模型支持单机8卡训练。
   - 单机8卡训练

     ```
     cd examples/aishell/s0/test
     bash train_full_8p.sh --stage=起始stage --stop_stage=终止stage --data_path=/data/xxx/  # 8卡精度
     bash train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     bash train_full_8p_whisper.sh --stage=起始stage --stop_stage=终止stage --data_path=/data/xxx/  # 裁剪了CNN Module的8卡精度、性能
     ```

   模型训练脚本参数说明如下。

   ```shell
   --stage              //模型训练的起始阶段，默认为-1，即从数据下载开始启动训练。若之前数据下载、准备、特征生成等阶段已完成，可配置--stage=4开始训练。
   --stop_stage         //模型训练的终止阶段
   --data_path          //数据集路径
   ```

   > **说明：**
   > 
   > --stage <-1 ~ 5>、--stop_stage <-1 ~ 5>：控制模型训练的起始、终止阶段。模型包含 -1 ~ 5 训练阶段，其中 -1 ~ 3 为数据下载、准备、特征生成等阶段，4为模型训练，5为ASR任务评估。首次运行时请从 -1 开始，-1 ~ 3 阶段执行过一次之后，后续可以从stage 4 开始训练。
   > 
   > --data_path参数填写数据集路径，需要写到数据集的一级目录。

   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

### Profiling日志采集（可选）

本仓已提供基于 Ascend Profiler 的简单 profiling 能力，通过环境变量控制开启时机和采集步数：

- `PROFILE_START_STEP`：从第几个 step 开始采集，只采集一个 step 的数据。
- `PROFILE_TYPE`：profiling 类型，目前支持 `ASCEND_PROFILER`。

使用示例（以单机 8 卡训练脚本为例）：

```bash
export PROFILE_START_STEP=5   # 从第 5 个 step 开始采集，只采集该 step
export PROFILE_TYPE=ASCEND_PROFILER

cd examples/aishell/s0/test
bash train_full_8p.sh --stage=起始stage --stop_stage=终止stage --data_path=/data/xxx/
```

常见说明：

- 如无 profiling 需求，可不设置上述环境变量，训练流程不受影响。
- 由于 profiling 会增加一定开销，建议在较短的训练阶段或少量 step 上使用，避免影响整体性能评估。


## 训练结果展示

**表 3**  conformer训练结果展示表

|    NAME     | Error | FPS(iters/sec) | Epochs | AMP_Type | Torch_Version |
|:-----------:| :---: |:--------------:|:------:|:--------:|:-------------:|
|   8p-竞品A    |   -   |     800.44     |   15   |   fp32   |     1.11      |
| 8p-Atlas 800T A2  |   -   |     526.34     |   15   |   fp32   |     1.11      |
|   8p-竞品A    |   -   |     958.98     |   15   |   fp32   |      2.1      |
| 8p-Atlas 800T A2  |   -   |     830.49     |   15   |   fp32   |      2.1      |

**表 4**  whisper训练结果展示表

|     NAME      | Error | FPS(iters/sec) | Epochs | AMP_Type | Torch_Version |
|:-------------:| :---: |:--------------:|:------:|:--------:|:-------------:|
|    8p-竞品A     |   -   |     746.39     |   15   |   fp32   |     1.11      |
|  8p-Atlas 800T A2   |   -   |     667.62     |   15   |   fp32   |     1.11      |
|   8p-竞品A      |   -   |     748.85     |   15   |   fp32   |      2.1      |
|  8p-Atlas 800T A2   |   -   |     789.31     |   15   |   fp32   |      2.1      |

**表 5** conformer result
* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 18, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode             | WER  |
|:------:|:----:|
| ctc greedy search        | 4.96 |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------ |:-------:|:------:|
| Conformer 8p-竞品 | FP32 | 958.98 |
| Conformer 8p-Atlas 900 A2 PoDc | FP32 | 1166.96 |
| whisper 8p-竞品 | FP16 | 748.85 |
| whisper 8p-Atlas 900 A2 PoDc | FP16 | 1470|

# 公网地址说明

代码涉及公网地址参考[public_address_statement.md](./public_address_statement.md)

# 变更说明

2023.09.01：首次发布。

2024.03.16: 增加PyTorch2.1基线，增加FAQ。

<a id="FAQ"></a>
# FAQ

Q1：Pytorch2.1版本，运行时可能出现段错误

A1：问题原因是Pytorch与torchaudio版本不匹配，需要手动编译安装torchaudio，并在编译时设置变量 BUILD_SOX=0

同时对模型内部分相关代码进行修改：

1). 在tools/compute_cmvn_stats.py中修改：
  ```shell
  # 1. 找到：
  torchaudio.set_audio_backend("sox_io")
  # 修改为：
  # torchaudio.set_audio_backend("sox_io")
  
  # 2. 找到：
  sample_rate = torchaudio.backend.sox_io_backend.info(wav_path).sample_rate
  # 修改为：
  sample_rate = torchaudio.info(wav_path).sample_rate
  
  # 3. 找到：
  waveform, sample_rate = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
  # 修改为：
  waveform, sample_rate = torchaudio.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
  ```
  
2). 在tools/make_shard_list.py中修改：
  ```shell
  # 1. 找到：
  import torchaudio.backend.sox_io_backend as sox
  # 修改为：
  # import torchaudio.backend.sox_io_backend as sox
  
  # 2. 找到：
  waveforms, sample_rate = sox.load(wav, normalize=False)
  # 修改为：
  waveforms, sample_rate = torchaudio.load(wav, normalize=False)
  
  # 3. 找到：
  sox.save(f, audio, resample, format="wav", bits_per_sample=16)
  # 修改为：
  torchaudio.save(f, audio, resample, format="wav", bits_per_sample=16)
  ```

2). 在wenet/dataset/processor.py中修改：
  ```shell
  # 1. 找到：
  torchaudio.utils.sox_utils.set_buffer_size(16500)
  # 修改为：
  # torchaudio.utils.sox_utils.set_buffer_size(16500)
  
  # 2. 找到：
  sample_rate = torchaudio.backend.sox_io_backend.info(
                    wav_file).sample_rate
  # 修改为：
  sample_rate = torchaudio.info(
                    wav_file).sample_rate
  
  # 3. 找到：
  waveform, _ = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
  # 修改为：
  waveform, _ = torchaudio.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
  
  # 4. 找到：
          if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav
  # 修改为：
          # if speed != 1.0:
          #   wav, _ = torchaudio.sox_effects.apply_effects_tensor(
          #       waveform, sample_rate,
          #       [['speed', str(speed)], ['rate', str(sample_rate)]])
          #   sample['wav'] = wav
  ```


