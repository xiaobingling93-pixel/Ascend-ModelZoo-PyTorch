## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持：
Atlas 800I A2推理设备：支持的卡数最小为1
- [Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
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

### 1.3 环境依赖安装
```shell
pip3 install -r requirements.txt
```

### 1.4 MindIE安装
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

### 1.5 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```
## 二、下载本仓库

### 2.1 下载到本地
```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

## 三、OpenSora1.2使用

### 3.1 权重及配置文件说明
1. text_encoder权重链接:
```shell
   https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main
```
2. tokenizer权重链接：
```shell
   https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main
```
3. STDiT3权重链接：
- 下载该权重，并重命名为transformer
```shell
   https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3/tree/main
```
- 修改该权重的config.json
```shell
   将enable_flash_attn设置为true
```
4. VAE权重链接：
- 下载该权重，并重命名为vae
```shell
https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2/tree/main
```
- 修改该权重的config.json
```shell
修改architectures和model_type字段为VideoAutoencoder即可。
```
5. VAE_2d：
- 权重链接如下, 下载后将vae_2d的配置文件和权重文件放在vae/vae_2d/vae路径下。
```shell
https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main
```
6. scheduler:
- 新增scheduler_config.json配置文件, 内容如下所示: 
```shell
{
    "_class_name": "RFlowScheduler",
    "_mindiesd_version": "1.0.0",
    "num_sampling_steps": 30,
    "num_timesteps": 1000
}
```
7.新增model_index.json
将以上步骤下载的权重放在同一目录下, 并新增model_index.json文件, 该文件内容如下所示
```shell
{
    "_class_name": "OpenSoraPipeline",
    "_mindiesd_version": "1.0.0",
    "scheduler": [
        "mindiesd",
        "RFlowScheduler"
    ],
    "text_encoder": [
        "transformers",
        "T5EncoderModel"
    ],
    "tokenizer": [
        "transformers",
        "AutoTokenizer"
      ],
    "transformer": [
        "mindiesd",
        "STDiT3"
    ],
    "vae": [
        "mindiesd",
        "VideoAutoencoder"
    ]
}
```
8.各模型的配置文件、权重文件的层级样例如下所示。
```commandline
|----open-sora
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
|    |    |---- vae_2d
|    |    |    |---- vae
|    |    |    |    |---- config.json
|    |    |    |    |---- 模型权重
```

### 3.2 单卡性能测试
设置权重路径
```shell
path = './path'
```
执行命令：
```shell
python inference_opensora12.py \
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

### 3.3 多卡性能测试
设置权重路径
```shell
path = './path'
```

执行命令：
```shell
torchrun --nproc_per_node=4 inference_opensora12.py \
       --path ${path} \
       --type bf16 \
       --num_frames 32 \
       --image_size 720,1280 \
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


## 四、优化指南
本模型使用的优化手段如下：
- 等价优化：FA、LayerNorm、DSP
- 算法优化：DitCache


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。