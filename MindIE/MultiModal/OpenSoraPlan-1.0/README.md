# MindIE SD

## 一、介绍
MindIE SD是MindIE的视图生成推理模型套件，其目标是为稳定扩散（Stable Diffusion, SD）系列大模型推理任务提供在昇腾硬件及其软件栈上的端到端解决方案，软件系统内部集成各功能模块，对外呈现统一的编程接口。

## 二、安装依赖

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
- [800I A2](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.0.RC1.beta1&driver=1.0.RC1.alpha)
- [Duo卡](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=17&cann=8.0.RC2.alpha002&driver=1.0.22.alpha)
2. [安装指导手册](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0019.html)
### 2.2 CANN开发套件包+kernel包+MindIE包下载
1. 下载：
- [800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [Duo卡](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
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

## 三、OpenSora1.2使用

### 3.1 权重及配置文件说明
#### 3.1.1 下载子模型权重
首先需要下载以下子模型权重和配置文件：text_encoder、tokenizer、transformer、vae、scheduler

1.text_encoder:
- 下载配置文件和权重文件，下载链接：
```shell
   https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main
```
下载后重命名文件夹为text_encoder

2.tokenizer:
将上述下载的text_encoder的权重和配置文件中的tokenizer_config.json和spiece.model拷贝并单独存放至另一个文件夹，重命名文件夹为tokenizer

3.transformer：
- 下载配置文件和权重文件，下载链接：
```shell
https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3/tree/main
```
下载后重命名文件夹为transformer

4.vae：
vae需要下载两部分权重:VAE和VAE_2d

(1) VAE
- 按以下链接下载权重和配置文件，并修改配置文件的architectures和model_type字段为VideoAutoencoder。参考MindIE-SD/examples/open-sora/vae/config.json。
```shell
https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2/tree/main
```
下载后重命名文件夹为vae

(2) VAE_2d：
- 按以下链接下载配置文件和权重文件，在上述vae文件夹下新建vae_2d/vae目录，并将下载的权重文件放置在路径下。
```shell
https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main
```
5.scheduler:
- 采样器无需权重文件，配置文件参考MindIE-SD/examples/open-sora/scheduler/scheduler_config.json设置，并放置在scheduler文件夹下。

#### 3.1.2 配置Pipeline
1. 新建model_index.json配置文件，参考MindIE-SD/examples/open-sora-plan/model_index.json，与其他子模型文件夹同级目录。并将整体Pipeline权重文件夹命名为OpenSora1.2。

2. 配置完成后示例如下。
```commandline
|----OpenSora1.2
|    |---- model_index.json
|    |---- scheduler
|    |    |---- scheduler_config.json
|    |---- text_encoder
|    |    |---- config.json
|    |    |---- pytorch_model-00001-of-00002.bin
|    |    |---- pytorch_model-00002-of-00002.bin
|    |    |---- pytorch_model.bin.index.json
|    |    |---- special_tokens_map.json
|    |    |---- spiece.model
|    |    |---- tokenizer_config.json
|    |---- tokenizer
|    |    |---- spiece.model
|    |    |---- tokenizer_config.json
|    |---- transformer
|    |    |---- config.json
|    |    |---- model.safetensors
|    |---- vae
|    |    |---- config.json
|    |    |---- model.safetensors
|    |    |---- vae_2d
|    |    |    |---- vae
|    |    |    |    |---- config.json
|    |    |    |    |---- diffusion_pytorch_model.safetensors
```

### 3.2 安装依赖库
进入MindIE-SD路径，安装MindIE-SD的依赖库。
```
pip install -r requirements.txt
```
安装colossalai。 colossalai0.4.4 版本会自动安装高版本torch, 所以要单独安装。
```
pip install colossalai==0.4.4 --no-deps
```
### 3.3 单卡性能测试
设置权重路径
```shell
path='./path'
```
执行命令：
```shell
python tests/inference_opensora12.py \
       --path ${path} \
       --device_id 0 \
       --type bf16 \
       --num_frames 32 \
       --image_size 720,1280 \
       --fps 8
```
参数说明：
- path: 权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。
- device_id: 推理设备ID。
- type: bf16、fp16。
- num_frames:总帧数，范围：32, 128。
- image_size：(720, 1280)、(512, 512)。
- fps: 每秒帧数：8。
- test_acc: 使用--test_acc开启全量视频生成，用于精度测试。性能测试时，不开启该参数。

### 3.4 多卡性能测试
设置权重路径
```shell
path='./path'
```

执行命令：
```shell
torchrun --nproc_per_node=4 tests/inference_opensora12.py \
       --path ${path} \
       --type bf16 \
       --num_frames 32 \
       --image_size (720,1280) \
       --fps 8 \
       --enable_sequence_parallelism True
```
参数说明： 
- nproc_per_node: 并行推理的总卡数。
- enable_sequence_parallelism 开启dsp 多卡并行
- path: 权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。
- type: bf16、fp16。
- num_frames:总帧数，范围：32, 128。
- image_size：(720, 1280)、(512, 512)。
- fps: 每秒帧数：8。

精度测试参考第五节Vbench精度测试。

## 四、OpenSoraPlan1.0使用

### 4.1 权重及配置文件说明
#### 4.1.1 下载子模型权重
首先需要下载以下子模型权重：text_encoder、tokenizer、transformer、vae

1.text_encoder:
- 下载配置文件和权重文件，下载链接：
```shell
https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main
```
下载后重命名文件夹为text_encoder

2.tokenizer:
将上述下载的text_encoder的权重和配置文件中的tokenizer_config.json和spiece.model拷贝并单独存放至另一个文件夹，重命名文件夹为tokenizer

3.transformer：
- 下载配置文件和权重文件，根据需要下载不同分辨率和帧数的权重和配置文件，当前支持17x256x256、65x256x256、65x512x512三种规格，选择一种规格下载即可。下载链接：
```shell
https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main
```
下载完成后重命名文件夹为transformer

4.vae：
- 下载配置文件和权重文件，下载该链接下的vae文件夹：
```shell
https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main
```

#### 4.1.2 配置Pipeline
1.新建model_index.json配置文件，参考MindIE-SD/examples/open-sora-plan/model_index.json，与其他子模型文件夹同级目录。并将整体Pipeline权重文件夹命名为Open-Sora-Plan-v1.0.0。

2.配置完成后示例如下。
```commandline
|----Open-Sora-Plan-v1.0.0
|    |---- model_index.json
|    |---- text_encoder
|    |    |---- config.json
|    |    |---- pytorch_model-00001-of-00002.bin
|    |    |---- pytorch_model-00002-of-00002.bin
|    |    |---- pytorch_model.bin.index.json
|    |    |---- special_tokens_map.json
|    |    |---- spiece.model
|    |    |---- tokenizer_config.json
|    |---- tokenizer
|    |    |---- spiece.model
|    |    |---- tokenizer_config.json
|    |---- transformer
|    |    |---- config.json
|    |    |---- diffusion_pytorch_model.safetensors
|    |---- vae
|    |    |---- config.json
|    |    |---- diffusion_pytorch_model.safetensors
```

### 4.2 安装依赖库
进入MindIE-SD/mindiesd/requirements路径，安装open-sora-plan1.0的依赖库。
```
pip install -r requirements_opensoraplan.txt
```
安装colossalai。 colossalai0.4.4 版本会自动安装高版本torch, 所以要单独安装。
```
pip install colossalai==0.4.4 --no-deps
```
### 4.3 单卡性能测试
设置权重路径
```shell
model_path='./model_path'
```
执行命令：
```shell
python tests/inference_opensora_plan.py \
       --model_path ${model_path} \
       --text_prompt tests/t2v_sora.txt \
       --sample_method PNDM \
       --save_img_path ./sample_videos/t2v_PNDM \
       --image_size 512 \
       --fps 24 \
       --guidance_scale 7.5 \
       --num_sampling_steps 250 \
       --seed 5464
```
参数说明：
- model_path: 权重路径，包含vae、text_encoder、Tokenizer、Transformer和Scheduler五个模型的配置文件及权重。
- text_prompt: 输入prompt，可以为list形式或txt文本文件（按行分割）。
- sample_method：采样器名称，默认PNDM，只支持['DDIM', 'EulerDiscrete', 'DDPM', 'DPMSolverMultistep','DPMSolverSinglestep', 'PNDM', 'HeunDiscrete', 'EulerAncestralDiscrete', 'DEISMultistep', 'KDPM2AncestralDiscrete']。若要使用采样步数优化，则选择"DPMSolverSinglestep"或"DDPM"。
- save_img_path：生成视频的保存路径，默认./sample_videos/t2v。
- image_size：生成视频的分辨率，需与下载的transformer权重版本对应，为512或256。
- fps：生成视频的帧率，默认24。
- guidance_scale：生成视频中cfg的参数，默认7.5。
- num_sampling_steps：生成视频采样迭代次数，默认250步。若采用"DPMSolverSinglestep"或"DDPM"采样器，可设置为50步。
- seed: 随机种子设置。

注：若出现"RuntimeError: NPU out of memory."报错，可能是torch_npu最新版本默认把虚拟内存关闭，可尝试设置```export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True```环境变量解决。

### 4.4 多卡性能测试
设置权重路径、并行推理的总卡数
```shell
model_path='./model_path'
NUM_DEVICES=4
```

执行命令：
```shell
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=$NUM_DEVICES tests/inference_opensora_plan.py \
       --model_path ${model_path} \
       --text_prompt tests/t2v_sora.txt \
       --sample_method PNDM \
       --save_img_path ./sample_videos/t2v_PNDM_dsp_$NUM_DEVICES \
       --image_size 512 \
       --fps 24 \
       --guidance_scale 7.5 \
       --num_sampling_steps 250 \
       --seed 5464 \
       --sequence_parallel_size $NUM_DEVICES
```
参数说明： 
- ASCEND_RT_VISIBLE_DEVICES: 指定使用的具体推理设备ID
- nproc_per_node: 并行推理的总卡数。
- sequence_parallel_size: 序列并行的数量。
其余参数同上

### 4.5 开启patch相似性压缩测试
设置权重路径
```shell
model_path='./model_path'
```
执行命令：
```shell
python tests/inference_opensora_plan.py \
       --model_path ${model_path} \
       --text_prompt tests/t2v_sora.txt \
       --sample_method PNDM \
       --save_img_path ./sample_videos/t2v_PNDM \
       --image_size 512 \
       --fps 24 \
       --guidance_scale 7.5 \
       --num_sampling_steps 250 \
       --use_cache \
       --cache_config 5,27,5,2 \
       --cfg_last_step 150 \
       --seed 5464
```
参数说明：
- use_cache: 是否开启DiT-Cache，不设置则不开启。
- cache_config: DiT-Cache的配置参数，需设置4个数值，分别为start_block_idx, end_block_idx, start_step, step_interval。
- cfg_last_step：开启跳过cfg计算的步数。
其余参数同上。

精度测试参考第五节Vbench精度测试。

## 五、Vbench精度测试(基于gpu)
1、视频生成完成后，精度测试推荐使用业界常用的VBench(Video Benchmark)工具，详见如下链接：
```shell
https://github.com/Vchitect/VBench
```
2、当前主要评估指标为[subject_consistency, imaging_quality, aesthetic_quality, overall_consistency, motion_smoothness]。

注：vbench各精度指标平均下降不超过1%可认为该视频质量无下降。
