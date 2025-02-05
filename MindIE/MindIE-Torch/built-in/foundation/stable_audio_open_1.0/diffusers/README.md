# stable-audio-open-1.0模型-diffusers方式推理指导

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

- 设备支持：
Atlas 800I A2推理设备：支持的卡数为1
Atlas 300I Duo推理卡：支持的卡数为1

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
   apt-get update
   apt-get install libsndfile1
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
   python3 diffusers_aie_patch.py
   python3 brownian_interval_patch.py
   ```

4. MindieTorch配套Torch_NPU使用

   MindieTorch采用dlopen的方式动态加载Torch_NPU，需要手动编译libtorch_npu_bridge.so，并将其放在libtorch_aie.so同一路径下，或者将其路径设置到LD_LIBRARY_PATH环境变量中，具体参考：
   ```bash
   https://www.hiascend.com/document/detail/zh/mindie/10RC2/mindietorch/Torchdev/mindie_torch0017.html
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   1. 提前下载权重，放到代码同级目录下。

       ```bash
       # 需要使用 git-lfs (https://git-lfs.com)
       git lfs install
       
       # 下载stable-audio-open-1.0权重
       git clone https://huggingface.co/stabilityai/stable-audio-open-1.0
       ```
   
   2. 导出pt模型并进行编译。

      (1) 设置模型权重的路径
      ```bash
      # stable-audio-open-1.0 (执行时下载权重)
      model_base="stabilityai/stable-audio-open-1.0"
      
      # stable-audio-open-1.0 (使用上一步下载的权重)
      model_base="./stable-audio-open-1.0"
      ```

      (2) 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         ```

      (3) 执行export命令
   
         ```bash
         python3 export_ts.py --model ${model_base} --output_dir ./models --soc Ascend${chip_name} --device 0
         ```

      参数说明：
         - --model：模型权重路径
         - --output_dir: 存放导出模型的路径
         - --soc：处理器型号。
         - --device：推理设备ID
      
      注意：trace+compile耗时较长且占用较多的CPU资源，请勿在执行export命令时运行其他占用CPU内存的任务，避免程序意外退出。
   
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
      numactl -C 0-23 python3 stable_audio_open_aie_pipeline.py \
              --model ${model_base} \
              --output_dir ./models \
              --prompt_file ./prompts.txt \
              --num_inference_steps 100 \
              --audio_end_in_s 10 10 47 \
              --num_waveforms_per_prompt 1 \
              --guidance_scale 7 \
              --save_dir ./results \
              --device 0
      ```
      
      参数说明：
      - --model：模型权重路径。
      - --output_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --num_inference_steps: 语音生成迭代次数。
      - --audio_end_in_s：生成语音的时长，如不输入则默认生成10s。
      - --num_waveforms_per_prompt：一个提示词生成的语音数量。
      - --guidance_scale：音频生成质量与准确度系数。
      - --save_dir：生成语音的存放目录。
      - --device：推理设备ID。
      
      执行完成后在`./results`目录下生成推理语音，语音生成顺序与文本中prompt顺序保持一致，并在终端显示推理时间。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
性能参考下列数据。

### Stable-Audio-Open-1.0

| 硬件形态 | 迭代次数 | 平均耗时|
| :------: |:----:|:----:|
| Atlas 800I A2(8*32G) |  100  |  5.895s  |