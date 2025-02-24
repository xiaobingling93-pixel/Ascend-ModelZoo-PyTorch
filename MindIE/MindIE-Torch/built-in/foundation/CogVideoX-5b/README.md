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
- 设备支持：
Atlas 800I A2/Atlas 800T A2设备：支持的卡数最小为1
- [Atlas 800I A2/Atlas 800T A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
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
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

## 三、CogVideoX-5b / CogVideoX-2b使用

### 3.1 权重及配置文件说明
1. 下载CogVideoX-5b / CogVideoX-2b权重:（scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重）
```shell
   git clone https://huggingface.co/THUDM/CogVideoX-5b
   git clone https://huggingface.co/THUDM/CogVideoX-2b
```
2. 各模型的配置文件、权重文件的层级样例如下所示。
```commandline
|----CogVideoX-5b / CogVideoX-2b
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

### 3.2 RoPE算子编译
进入算子路径，执行编译命令
```shell
cd pta_plugin
bash build.sh
```
编译成功后会在build文件夹下生成.so结尾的算子文件


在cogvideox_5b/models/attention_processor.py脚本中添加编译生成的算子路径
```python
torch.ops.load_library("./pta_plugin/build/libPTAExtensionOPS.so")
```
注意：首次运行需要加载RoPE算子，请在正式推理前进行warmup

### 3.3 单卡单prompt功能测试
1. 设置CogVideoX-5b权重路径：
```shell
model_path='data/CogVideoX-5b'
```

或者设置CogVideoX-2b权重路径：
```shell
model_path='data/CogVideoX-5b'
```

2. 执行命令：
```shell
export CPU_AFFINITY_CONF=1
export HCCL_OP_EXPANSION_MODE="AIV"
TASK_QUEUE_ENABLE=2 ASCEND_RT_VISIBLE_DEVICES=0 torchrun --master_port=2002 --nproc_per_node=1 inference.py \
        --prompt_file ./prompts.txt \
        --model_path ${model_path} \
        --output_path ./output \
        --num_frames 48 \
        --width 720 \
        --height 480 \
        --fps 8 \
        --num_inference_steps 50 \
        --dtype bfloat16 \
        --seed 42 \
        --enable_skip
```
参数说明：
- CPU_AFFINITY_CONF=1：环境变量，绑核。
- HCCL_OP_EXPANSION_MODE="AIV"：环境变量，通信算子编排。
- TASK_QUEUE_ENABLE=2：开启二级流水。
- ASCEND_RT_VISIBLE_DEVICES=0：device id，可设置其他卡数。
- prompt_file：文本文件，用于视频生成的文字描述提示。
- model_path：权重路径，包含scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重。
- output_path：生成视频的保存路径。
- num_frames：生成视频的帧数，默认值为48。
- width：生成视频的分辨率，宽，默认值为720。
- height：生成视频的分辨率，高，默认值为480。
- fps：生成视频的帧率，默认值为8。
- num_inference_steps：推理迭代步数，默认值为50。
- dtype：数据类型，默认值为bfloat16。CogVideoX-2b推荐设置为float16，需要在命令前加INF_NAN_MODE_FORCE_DISABLE=1，开启饱和模式避免数值溢出。
- seed: 设置随机种子，默认值为42。
- enable_skip：是否使用采样优化。
推理结束后会在当前路径下生成result.json，用于记录文本提示和生成视频的对应关系，便于测试视频精度。


## 四、推理性能结果参考
### CogVideoX-5b
| 硬件形态  | cpu规格 | batch size | 迭代次数 | 数据类型 | 平均耗时 |
| :------: | :------: | :------: |:----:| :------: | :------: |
| Atlas 800I A2(8*64G) | 64核(arm) |  1  |  50  | bfloat16 | 240s |

### CogVideoX-2b
| 硬件形态  | cpu规格 | batch size | 迭代次数 | 数据类型 | 平均耗时 |
| :------: | :------: | :------: |:----:| :------: | :------: |
| Atlas 800I A2(8*64G) | 64核(arm) |  1  |  50  | float16 | 102s |

性能测试需要独占npu和cpu

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。