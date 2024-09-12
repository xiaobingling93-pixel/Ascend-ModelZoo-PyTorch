# open-stable-audio-1.0模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)
  

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   [此处获得](https://huggingface.co/stabilityai/stable-audio-open-1.0)

- 参考实现：
  ```bash
   # StableAudioOpen1.0
   https://huggingface.co/stabilityai/stable-audio-open-1.0
  ```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 获取源码<a name="section4622531142816"></a>
1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt
   apt-get install libsudfile1
   ```

2. 安装mindie包

   ```bash
   # 安装mindie
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```
   
3. 代码修改

- 执行命令：
   ```bash
   python3 stable_audio_open_attention_processor_path.py
   python3 stable_audio_open_brownian_interval_path.py
   ```

## 模型推理<a name="section741711594517"></a>

1. 获取权重。

   1. 提前下载权重，放到代码同级目录下。

       ```bash
       # 需要使用 git-lfs (https://git-lfs.com)
       git lfs install
       
       # 下载stable-audio-open-1.0权重
       git clone https://huggingface.co/stabilityai/stable-audio-open-1.0
       ```
   
2. 开始推理验证。

   1. 开启cpu高性能模式
      ```bash
      echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
      sysctl -w vm.swappiness=0
      sysctl -w kernel.numa_balancing=0
      ```

   2. 安装绑核工具
      ```bash
      apt-get update
      apt-get install numactl
      ```
      查询卡的NUMA node
      ```shell
      lspci -vs bus-id
      ```
      bus-id可通过npu-smi info获得，查询到NUMA node，在推理命令前加上对应的数字

      可通过lscpu获得NUMA node对应的CPU核数
      ```shell
      NUMA node0: 0-23
      NUMA node1: 24-47
      NUMA node2: 48-71
      NUMA node3: 72-95
      ```
      当前查到NUMA node是0，对应0-23，推荐绑定其中单核以获得更好的性能。
   
   3. 执行推理脚本。
      ```bash
      export model_path=下载模型路径
      numactl -C 0-23 python3 stable_audio_open_pipeline.py \
              --stable_audio_open_dir ${model_path} \
              --prompt_file ./prompts.txt \
              --num_inference_steps 100 \
              --audio_end_in_s 10 10 47 \
              --num_waveforms_per_prompt 1 \
              --guidance_scale 7 \
              --device 0 \
              --save_dir ./result
      ```
      
      参数说明：
      - --stable_audio_open_dir：模型权重路径。
      - --prompt_file：提示词文件。
      - --num_inference_steps: 语音生成迭代次数。
      - --save_dir：生成语音的存放目录。
      - --device：推理设备ID。
      - --audio_end_in_s：生成语音的时长，如不输入则默认生成10s。
      - --num_waveforms_per_prompt：一个提示词生成的语音数量。
      - --guidance_scale：音频生成质量与准确度系数。
      
      执行完成后在`./results`目录下生成推理语音，语音生成顺序与文本中prompt顺序保持一致，并在终端显示推理时间。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
