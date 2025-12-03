# 模型推理指导  

## 一、模型简介

stable-audio-open-1.0是一种文本到语音的扩散模型，能够在给定任何文本输入的情况下生成相符的语音。参考实现请查看[stable-audio-open-1.0 blog](https://huggingface.co/stabilityai/stable-audio-open-1.0)。

本模型使用的优化手段如下：
- 等价优化：FA、RoPE、Linear
- 算法优化：DiTcache

## 二、环境准备

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- 支持卡数：Atlas 800I A2支持的卡数为1；Atlas 300I Duo支持的卡数为1
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)
**注意**：该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

### 2.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 2.4 MindieTorch配套Torch_NPU使用
MindieTorch采用dlopen的方式动态加载Torch_NPU，需要手动编译libtorch_npu_bridge.so，并将其放在libtorch_aie.so同一路径下，或者将其路径设置到LD_LIBRARY_PATH环境变量中，具体参考：
```bash
https://www.hiascend.com/document/detail/zh/mindie/100/mindietorch/Torchdev/mindie_torch0018.html
```

### 2.5 下载本仓库
```shell
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
```

### 2.6 安装所需依赖
按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
```shell
pip install -r requirements.txt
apt-get update
apt-get install libsndfile1
```

## 三、模型权重
```bash
# 需要使用 git-lfs (https://git-lfs.com)
git lfs install

# 下载stable-audio-open-1.0权重
git clone https://huggingface.co/stabilityai/stable-audio-open-1.0
```

## 四、模型推理

### 4.1 代码修改
```bash
python3 diffusers_aie_patch.py
python3 brownian_interval_patch.py
```

### 4.2 模型转换
使用Pytorch导出pt模型，再使用MindIE推理引擎转换为适配昇腾的模型。

1. 设置权重路径：
```bash
# stable-audio-open-1.0 (执行时下载权重)
model_base="stabilityai/stable-audio-open-1.0"

# stable-audio-open-1.0 (使用上一步下载的权重)
model_base="./stable-audio-open-1.0"
```

2. 查看芯片名称（$\{chip\_name\}）
```
npu-smi info
```

3. 执行命令
```bash
python3 export_ts.py --model ${model_base} --output_dir ./models --soc Ascend${chip_name} --device 0
```

参数说明：
- --model：模型权重路径
- --output_dir: 存放导出模型的路径
- --soc：处理器型号。
- --device：推理设备ID

注意：trace+compile耗时较长且占用较多的CPU资源，请勿在执行export命令时运行其他占用CPU内存的任务，避免程序意外退出。
   
### 4.3 性能测试

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

## 五、推理结果参考
### Stable-Audio-Open-1.0 性能 & 精度数据

| 硬件形态 | 迭代次数 | 数据类型 | 性能|
| :------: |:----:|:----:|:----:|
| Atlas 800I A2(8*32G) |  100  | float16 |  5.895s  |

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。