# 模型推理指导  

- ## 注意
本模型仓代码，是针对此开源链接进行的适配
https://github.com/Tencent-Hunyuan/HunyuanVideo

## 一、模型简介

HunyuanVideo是一种文本到视频的扩散模型，能够在给定文本输入的情况下生成相符的视频。

  **表 1**  本模型当前支持的分辨率

  | 分辨率 | h/w=9:16 | h/w=9:16 | h/w=4:3 | h/w=3:4 | h/w=1:1 |
  | ---- | ---- | ---- | ---- | ---- | ---- |
  | 720P | 720x1280 | 1280x720 | 1104x832 | 832x1104 | 960x960 |

- 本模型使用的优化手段如下：
等价优化：FA、ROPE、RmsNorm、SP并行
算法优化：FA、ROPE、RmsNorm、SP并行、cache

## 二、环境准备

  **表 2**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10/3.11 | - |
  | torch | 2.1.0 | - |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- 支持卡数：支持的卡数：1、2、3、4、6、8、16
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

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

### 2.4 Torch_npu安装
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
pytorchversion即torch的版本

### 2.5 下载本仓库
```shell
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/MindIE/MultiModal/HunyuanVideo/
```

### 2.6 安装所需依赖
```shell
pip install -r requirements.txt
```

## 三、模型权重

### 3.1 权重下载
1. 权重链接
```shell
# text_encoder权重链接
https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers

# text_encoder_2权重链接
https://huggingface.co/openai/clip-vit-large-patch14

# hunyuan-model权重链接
https://huggingface.co/tencent/HunyuanVideo
```

权重目录如下所示：
```shell
HunyuanVideo
  ├──README.md
  ├──hunyuan-video-t2v-720p
  │  ├──transformers
  │  ├──vae
  ├──llava-llama-3-8b-v1_1-transformers
  ├──clip-vit-large-patch14
```

2. 权重修改
修改text_encoder的权重
```shell
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir llava-llama-3-8b-v1_1-transformers --output_dir text_encoder
```
修改之后的权重目录如下所示：
```shell
HunyuanVideo
  ├──README.md
  ├──hunyuan-video-t2v-720p
  │  ├──transformers
  │  ├──vae
  ├──text_encoder
  ├──clip-vit-large-patch14
```

## 四、模型推理

### 4.1 单卡等价优化推理性能测试
执行命令：
```shell
export TOKENIZERS_PARALLELISM=false
export ALGO=0
python sample_video.py \
      --model-base HunyuanVideo \
      --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
      --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
      --text-encoder-path HunyuanVideo/text_encoder \
      --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
      --model-resolution "720p" \
      --video-size 720 1280 \
      --video-length 129 \
      --infer-steps 50 \
      --prompt "A cat walks on the grass, realistic style." \
      --seed 42 \
      --flow-reverse \
      --num-videos 1 \
      --device_id 0 \
      --save-path ./results
```
参数说明：
- model-base: 权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。
- dit-weight: dit的权重路径
- vae-path: VAE的权重路径
- text-encoder-path: text_encoder的权重路径
- text-encoder-2-path: text_encoder_2的权重路径
- model-resolution: 分辨率
- video-size: 生成视频的高和宽
- video-length: 总帧数
- infer-steps: 推理步数
- prompt: 文本提示词
- seed: 随机种子
- num-videos: 每个prompt生成多少个视频，该参数和batch有关，800I A2(64G)机器上，该参数的大小受显存限制
- device_id：单卡推理时，可设置NPU id
- save-path: 生成的视频的保存路径
- flow-reverse：是否进行反向采样

### 4.2 单卡算法优化推理性能测试
执行命令：
```shell
export TOKENIZERS_PARALLELISM=false
export ALGO=0
python sample_video.py \
      --model-base HunyuanVideo \
      --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
      --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
      --text-encoder-path HunyuanVideo/text_encoder \
      --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
      --model-resolution "720p" \
      --video-size 720 1280 \
      --video-length 129 \
      --infer-steps 50 \
      --prompt "A cat walks on the grass, realistic style." \
      --seed 42 \
      --flow-reverse \
      --num-videos 1 \
      --device_id 0 \
      --use_cache \
      --use_cache_double \
      --use-cpu-offload \
      --save-path ./results
```
参数说明：
- model-base: 权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。
- dit-weight: dit的权重路径
- vae-path: VAE的权重路径
- text-encoder-path: text_encoder的权重路径
- text-encoder-2-path: text_encoder_2的权重路径
- model-resolution: 分辨率
- video-size: 生成视频的高和宽
- video-length: 总帧数
- infer-steps: 推理步数
- prompt: 文本提示词
- seed: 随机种子
- num-videos: 每个prompt生成多少个视频，该参数和batch有关，800I A2(64G)机器上，该参数的大小受显存限制
- device_id：单卡推理时，可设置NPU id
- use_cache: 使能单流block算法优化
- use_cache_double: 使能双流block算法优化
- save-path: 生成的视频的保存路径
- flow-reverse：是否进行反向采样
- use-cpu-offload：是否开启cpu负载均衡

### 4.3 8卡等价优化推理性能测试
执行命令：
```shell
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0
torchrun --nproc_per_node=8 sample_video.py \
      --model-base HunyuanVideo \
      --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
      --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
      --text-encoder-path HunyuanVideo/text_encoder \
      --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
      --model-resolution "720p" \
      --video-size 720 1280 \
      --video-length 129 \
      --infer-steps 50 \
      --prompt "A cat walks on the grass, realistic style." \
      --seed 42 \
      --flow-reverse \
      --ulysses-degree 8 \
      --ring-degree 1 \
      --vae-parallel \
      --num-videos 1 \
      --save-path ./results
```
参数说明： 
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- nproc_per_node: 并行推理的总卡数。
- model-base: 权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。
- dit-weight: dit的权重路径
- vae-path: VAE的权重路径
- text-encoder-path: text_encoder的权重路径
- text-encoder-2-path: text_encoder_2的权重路径
- model-resolution: 分辨率
- video-size: 生成视频的高和宽
- video-length: 总帧数
- infer-steps: 推理步数
- prompt: 文本提示词
- seed: 随机种子
- vae-parallel: vae部分使能并行，目前只支持8卡、16卡并行时使用
- num-videos: 每个prompt生成多少个视频，该参数和batch有关，800I A2(64G)机器上，该参数的大小受显存限制
- save-path: 生成的视频的保存路径
- flow-reverse：是否进行反向采样
- ulysses-degree：ulysses并行使用的卡数
- ring-degree: ring并行使用的卡数

### 4.4 8卡算法优化推理性能测试

使用attentioncache
执行命令：
```shell
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=0
torchrun --nproc_per_node=8 sample_video.py \
      --model-base HunyuanVideo \
      --dit-weight HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
      --vae-path HunyuanVideo/hunyuan-video-t2v-720p/vae \
      --text-encoder-path HunyuanVideo/text_encoder \
      --text-encoder-2-path HunyuanVideo/clip-vit-large-patch14 \
      --model-resolution "720p" \
      --video-size 720 1280 \
      --video-length 129 \
      --infer-steps 50 \
      --prompt "A cat walks on the grass, realistic style." \
      --seed 42 \
      --flow-reverse \
      --ulysses-degree 8 \
      --ring-degree 1 \
      --vae-parallel \
      --use_attentioncache \
      --num-videos 1 \
      --save-path ./results
```
参数说明： 
- ALGO: 为0表示默认FA算子；设置为1表示使用高性能FA算子
- nproc_per_node: 并行推理的总卡数。
- model-base: 权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。
- dit-weight: dit的权重路径
- vae-path: VAE的权重路径
- text-encoder-path: text_encoder的权重路径
- text-encoder-2-path: text_encoder_2的权重路径
- model-resolution: 分辨率
- video-size: 生成视频的高和宽
- video-length: 总帧数
- infer-steps: 推理步数
- prompt: 文本提示词
- seed: 随机种子
- vae-parallel: vae部分使能并行
- use_attentioncache: 使能attentioncache策略
- num-videos: 每个prompt生成多少个视频，该参数和batch有关，800I A2(64G)机器上，该参数的大小受显存限制
- save-path: 生成的视频的保存路径
- flow-reverse：是否进行反向采样
- ulysses-degree：ulysses并行使用的卡数
- ring-degree: ring并行使用的卡数


## 五、推理结果参考
### HunyuanVideo精度数据
使用prompts.txt测试了seed42-46五组种子的视频，并测试了vbench并取平均值，6个指标如下：
| 分辨率h*w | dynamic_degree | subject_consistency | imaging_quality | aesthetic_quality | overall_consistency |  motion_smoothness |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 720*1280 | 0.1516 | 0.9774 | 0.5283 | 0.6048 | 0.291 | 0.9931 |

## 注意
在并行场景下，使用算法优化时，使用attention_cache有爆显存风险，我们建议在不同场景使用不同的算法优化：
| 并行度 | 参数配置 |
| ---- | ---- |
| 2    | --use_cache --use_cache_double |
| 3    | --use_cache --use_cache_double |
| 4    | --use_cache --use_cache_double |
| 8    | --use_attentioncache |
| 16   | --use_attentioncache |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。