---
license: apache-2.0
---



# Opensoraplan1.3

## 一、介绍
此仓库是开源模型Opensoraplan1.3， 基于MindIE SD 的实现。 可以实现更高效的推理性能。运行此仓库代码，需要安装MindIE SD 及其依赖。


## 二、安装依赖

MindIE SD是MindIE的视图生成推理模型套件，其目标是为稳定扩散（Stable Diffusion, SD）系列大模型推理任务提供在昇腾硬件及其软件栈上的端到端解决方案，软件系统内部集成各功能模块，对外呈现统一的编程接口。

MindIE-SD其依赖组件为driver驱动包、firmware固件包、CANN开发套件包、推理引擎MindIE包，使用MindIE-SD前请提前安装这些依赖。

| 简称            | 安装包全名                                                                     | 默认安装路径                               | 版本约束                              |
| --------------- |---------------------------------------------------------------------------|--------------------------------------|-----------------------------------|
| driver驱动包    | 昇腾310P处理器对应驱动软件包：Ascend-hdk-310p-npu-driver_\{version\}\_{os}\-{arch}.run | /usr/local/Ascend                    | 24.0.rc1及以上                       |
| firmware固件包  | 昇腾310P处理器对应固件软件包：Ascend-hdk-310p-npu-firmware_\{version\}.run             | /usr/local/Ascend                    | 24.0.rc1及以上                       |
| CANN开发套件包   | Ascend-cann-toolkit\_{version}_linux-{arch}.run                           | /usr/local/Ascend/ascend-toolkit/latest | 8.0.RC1及以上                        |
| 推理引擎MindIE包 | Ascend-mindie\_\{version}_linux-\{arch}.run                               | /usr/local/Ascend/mindie/latest      | 和mindietorch严格配套使用                |
| torch           | Python的whl包：torch-{version}-cp310-cp310-{os}_{arch}.whl                   | -                                    | Python版本3.10.x，torch版本支持2.1.0 |

- {version}为软件包版本
- {os}为系统名称，如Linux
- {arch}为架构名称，如x86_64

### 2.1 安装驱动和固件

1. 获取地址
- [Atlas 800I A2(8*64G)](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.0.RC1.beta1&driver=1.0.RC1.alpha)
2. [安装指导手册](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0019.html)
### 2.2 CANN开发套件包+kernel包+MindIE包下载
1. 下载：
- [Atlas 800I A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
2. [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

3. 快速安装：
- CANN开发套件包+kernel包安装
```commandline
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
- MindIE包安装
```commandline
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

- MindIE SD不需要单独安装，安装MindIE时将会自动安装
- torch_npu 安装:
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```commandline
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

### 2.3 pytorch框架(支持版本为：2.1.0)
[安装包下载](https://download.pytorch.org/whl/cpu/torch/)

使用pip安装
```shell
# {version}表示软件版本号，{arch}表示CPU架构。
pip install torch-${version}-cp310-cp310-linux_${arch}.whl
```

### 2.4 安装依赖库
安装MindIE-SD的依赖库。
```
pip install -r requirements.txt
```

## 三、Opensoraplan1.3

### 3.1 权重及配置文件说明

1. text_encoder和tokenizer:
- 配置文件和权重文件
```shell
https://huggingface.co/google/mt5-xxl
```
2. transformer：
- 配置文件和权重文件
```shell
   https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/any93x640x640
```
3. VAE：
- 配置文件和权重文件
```shell
https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main
```

### 3.2 执行推理脚本
```shell
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3  torchrun --nnodes=1 --nproc_per_node 4 --master_port 29516 \
    inference_opensoraplan13.py \
    --model_path /path/to/transformer/ \
    --num_frames 93 \
    --height 640 \
    --width 640 \
    --text_encoder_name_1 "/path/to/text/encoder" \
    --text_prompt prompt.txt \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "/home/liuyaofu/planweight/vae" \
    --save_img_path "./video/save/path" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --rescale_betas_zero_snr \
    --prediction_type "v_prediction" \
    --save_memory \
    --sp \
    --use_cache
```
ASCEND_RT_VISIBLE_DEVICES 指定特定的NPU进行计算
--nproc_per_node 控制总NPU卡数进行计算

--model_path 指定transformers(DiT)模型权重配置路径， 下面包含config文件和权重文件
--num_frames 设置生成的总帧数
--height 设置输出图像的高度为多少像素
--width 设置输出图像的宽度为多少像素
--text_encoder_name_1 指定text_encoder权重配置路径
--text_prompt 指定输入的文本提示, 可以是一个txt文件或者一个prompt字符
--ae VAE的对视频的压缩规格
--ae_path 指定VAE模型权重配置路径
--save_img_path 指定视频保存的路径
--fps 设置帧率
--guidance_scale 设置引导比例,用于控制negative文本对视频生成的影响程度
--num_sampling_steps 设置采样步骤的数量
--max_sequence_length 512 设置prompt的最大长度为512
--num_samples_per_prompt 1 设置每个提示生成的样本数为1
--rescale_betas_zero_snr schedular 的配置
--prediction_type  schedular 的配置
--save_memory 运行VAE时尽量节省内存, 当生成视频较大时,要开启
--sp 是否开启序列并行
--use_cache 是否开启dit cache算法


## 四、reference

### 4.1 EulerAncestralDiscreteScheduler

本项目中的 `EulerAncestralDiscreteScheduler` 是从 [Hugging Face diffusers 库](https://github.com/huggingface/diffusers) 中引用的 `EulerAncestralDiscreteScheduler`。diffusers 库是一个用于生成扩散模型（diffusion models）的工具库，它提供了多种调度器（schedulers）来控制扩散过程。

在 `EulerAncestralDiscreteScheduler` 中，使用了 "linspace", "leading", 和 "trailing" 这几个概念。这些概念在下述论文中有所描述。

- 链接：[https://arxiv.org/abs/2305.08891](https://arxiv.org/abs/2305.08891)


### 许可证

本项目遵循 Apache License 2.0。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。


