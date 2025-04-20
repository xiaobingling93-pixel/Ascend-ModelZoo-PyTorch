## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I A2推理设备：支持的卡数为1
Atlas 300I Duo推理卡：支持的卡数为1
- [Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
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


### 1.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.4 Torch_npu安装
安装pytorch框架 版本2.1.0
[安装包下载](https://download.pytorch.org/whl/cpu/torch/)

使用pip安装
```shell
# {version}表示软件版本号，{arch}表示CPU架构。
pip install torch-${version}-cp310-cp310-linux_${arch}.whl
```
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

## 二、下载本仓库

### 2.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/stable_audio_open_1.0.git
```

### 2.2 依赖安装
```bash
pip3 install -r requirements.txt
apt-get update
apt-get install libsndfile1
```

## 三、Stable-Audio-Open-1.0 使用

### 3.1 权重及配置文件说明
stable-audio-open-1.0权重链接:
```shell
https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main
```

### 3.2 单卡功能测试
设置权重路径
```shell
model_base='./stable-audio-open-1.0'
```
执行命令：
```shell
# 不使用DiTCache策略
python3 inference_stableaudio.py \
        --model ${model_base} \
        --prompt_file ./prompts/prompts.txt \
        --num_inference_steps 100 \
        --audio_end_in_s 10 10 47 \
        --save_dir ./results \
        --seed 1 \
        --device 0

# 使用DiTCache策略
python3 inference_stableaudio.py \
        --model ${model_base} \
        --prompt_file ./prompts/prompts.txt \
        --num_inference_steps 100 \
        --audio_end_in_s 10 10 47 \
        --save_dir ./results \
        --seed 1 \
        --device 0 \
        --use_ditcache

# 使用AttentionCache策略
python3 inference_stableaudio.py \
        --model ${model_base} \
        --prompt_file ./prompts/prompts.txt \
        --num_inference_steps 100 \
        --audio_end_in_s 10 10 47 \
        --save_dir ./results \
        --seed 1 \
        --device 0 \
        --use_attentioncache
```
参数说明：
- --model：模型权重路径。
- --prompt_file：提示词文件。
- --num_inference_steps: 语音生成迭代次数。
- --audio_end_in_s：生成语音的时长，如不输入则默认生成10s。
- --save_dir：生成语音的存放目录。
- --seed：设置随机种子，不指定时默认使用随机种子。
- --device：推理设备ID。
- --use_ditcache: 【可选】使用DiTCache策略。
- --use_attentioncache: 【可选】使用AttentionCache策略。

执行完成后在`./results`目录下生成推理语音，语音生成顺序与文本中prompt顺序保持一致，并在终端显示推理时间。

### 3.2 模型推理性能

性能参考下列数据。

| 硬件形态 | 迭代次数 | 平均耗时（w/o Cache）| 平均耗时（with DiTCache）| 平均耗时（with AttentionCache）|
| :------: |:----:|:----:|:----:|:----:|
| Atlas 800I A2(8*32G) |  100  | 5.846s  | 4.847s | 5.282s |

## 五、优化指南
本模型使用的优化手段如下：
- 等价优化：FA、RoPE、Linear
- 算法优化：DiTcache、Attentioncache


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。