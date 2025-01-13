---
license: apache-2.0
frameworks:
  - PyTorch
language:
  - en
hardwares:
  - NPU
---
## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- [800I A2/800T A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
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

### 1.5 安装所需依赖。
```shell
pip3 install -r requirements.txt
```

## 二、下载本仓库

### 2.1 下载到本地
```shell
   git clone https://modelers.cn/MindIE/CogVideoX-5b.git
```

## 三、CogVideoX-5b使用

### 3.1 权重及配置文件说明
1. 下载CogVideoX-5b权重:（scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重）
```shell
   git clone https://modelers.cn/AI-Research/CogVideoX-5B.git
```
2. 各模型的配置文件、权重文件的层级样例如下所示。
```commandline
|----CogVideoX-5b
|    |---- model_index.json
|    |---- scheduler
|    |    |---- scheduler_config.json
|    |---- text_encoder
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- tokenizer
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- transformer
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- vae
|    |    |---- config.json
|    |    |---- 模型权重
```

### 3.2 单卡单prompt功能测试
设置权重路径：
```shell
model_path='data/CogVideoX-5b'
```

执行命令：
```shell
export CPU_AFFINITY_CONF=1
export HCCL_OP_EXPANSION_MODE="AIV"
TASK_QUEUE_ENABLE=2 ASCEND_RT_VISIBLE_DEVICES=0 torchrun --master_port=2002 --nproc_per_node=1 inference.py\
        --prompt "A dog" \
        --model_path ${model_path} \
        --num_frames 48 \
        --width 720 \
        --height 480 \
        --fps 8 \
        --num_inference_steps 50
```
参数说明：
- CPU_AFFINITY_CONF=1：环境变量，绑核。
- HCCL_OP_EXPANSION_MODE="AIV"：环境变量，通信算子编排。
- TASK_QUEUE_ENABLE=2：开启二级流水。
- ASCEND_RT_VISIBLE_DEVICES=0：device id，可设置其他卡数。
- prompt：用于视频生成的文字描述提示。
- model_path：权重路径，包含scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重。
- num_frames：生成视频的帧数。
- width：生成视频的分辨率，宽。
- height：生成视频的分辨率，高。
- fps：生成视频的帧率，默认值为8。
- num_inference_steps：推理迭代步数，默认值为50。
