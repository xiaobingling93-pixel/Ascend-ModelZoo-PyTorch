## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- [800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [Duo卡](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
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
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

## 三、HunyuanDiT使用

### 3.1 权重及配置文件说明
1. text_encoder权重链接:
```shell
   https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tree/main/text_encoder
```
2. text_encoder_2权重链接：
```shell
   https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tree/main/text_encoder_2
```
3. tokenizer权重链接：
```shell
   https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tree/main/tokenizer
```
4. tokenizer_2权重链接：
```shell
   https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tree/main/tokenizer_2
```
5. transformer权重链接：
```shell
   https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2/tree/main/t2i/model
```
- 修改该权重的config.json
```shell
{
  "architectures": [
    "HunyuanDiT2DModel"
  ],
  "input_size": [
    null,
    null
  ],
  "patch_size": 2,
  "in_channels": 4,
  "hidden_size": 1408,
  "depth": 40,
  "num_heads": 16,
  "mlp_ratio": 4.3637,
  "text_states_dim": 1024,
  "text_states_dim_t5": 2048,
  "text_len": 77,
  "text_len_t5": 256
}
```
6. vae权重链接：
```shell
   https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tree/main/vae
```
- 修改该权重的config.json
```shell
{
  "architectures": [
    "AutoencoderKL"
  ],
  "in_channels": 3,
  "out_channels": 3,
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D"
  ],
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "block_out_channels": [
    128,
    256,
    512,
    512
  ],
  "layers_per_block": 2,
  "act_fn": "silu",
  "latent_channels": 4,
  "norm_num_groups": 32,
  "sample_size": 512,
  "scaling_factor": 0.13025,
  "shift_factor": null,
  "latents_mean": null,
  "latents_std": null,
  "force_upcast": false,
  "use_quant_conv": true,
  "use_post_quant_conv": true
}
```
7. scheduler:
- 新增scheduler_config.json配置文件, 内容如下所示: 
```shell
{
  "_class_name": "DDPMScheduler",
  "_mindiesd_version": "1.0.0",
  "steps_offset": 1,
  "beta_start": 0.00085,
  "beta_end": 0.02,
  "num_train_timesteps": 1000
}
```
8. 新增model_index.json
将以上步骤下载的权重放在同一目录下, 并新增model_index.json文件, 该文件内容如下所示
```shell
{
    "_class_name": "HunyuanDiTPipeline",
    "_mindiesd_version": "1.0.RC3",
    "scheduler": [
      "mindiesd",
      "DDPMScheduler"
    ],
    "text_encoder": [
      "transformers",
      "BertModel"
    ],
    "text_encoder_2": [
      "transformers",
      "T5EncoderModel"
    ],
    "tokenizer": [
      "transformers",
      "BertTokenizer"
    ],
    "tokenizer_2": [
      "transformers",
      "T5Tokenizer"
    ],
    "transformer": [
      "mindiesd",
      "HunyuanDiT2DModel"
    ],
    "vae": [
      "mindiesd",
      "AutoencoderKL"
    ]
}
```
9. 各模型的配置文件、权重文件的层级样例如下所示。
```commandline
|----hunyuandit
|    |---- model_index.json
|    |---- scheduler
|    |    |---- scheduler_config.json
|    |---- text_encoder
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- text_encoder_2
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- tokenizer
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- tokenizer_2
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
设置权重路径
```shell
path="ckpts/hydit"
```
执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --prompt "青花瓷风格，一只小狗" \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 25
```
参数说明：
- path：权重路径，包含scheduler、text_encoder、text_encoder_2、tokenizer、 tokenizer_2、transformer、vae，七个模型的配置文件及权重。
- device_id：推理设备ID。
- prompt：用于图像生成的文字描述提示。
- input_size：需要生成的图像尺寸。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数。

### 3.3 单卡多prompts进行性能/精度测试
设置权重路径
```shell
path="ckpts/hydit"
```
执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --test_acc \
       --prompt_list "prompts/example_prompts.txt" \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 25
```
参数说明：
- path：权重路径，包含scheduler、text_encoder、text_encoder_2、tokenizer、 tokenizer_2、transformer、vae，七个模型的配置文件及权重。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启全量图像生成，用于性能/精度测试。单prompt功能测试时，不开启该参数。
- prompt_list：用于图像生成的文字描述提示的列表文件路径。
- input_size：需要生成的图像尺寸。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数。

### 3.4 用LoRA进行测试
设置权重路径
```shell
path="ckpts/hydit"
```
LoRA权重链接：
```shell
   https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA/tree/main
```
设置LoRA权重路径
```shell
lora_path = 'ckpts/lora'
```
执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --prompt "青花瓷风格，一只小狗" \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 25
       --use_lora \
       --lora_ckpt ${lora_path}
```
参数说明：
- path：权重路径，包含scheduler、text_encoder、text_encoder_2、tokenizer、 tokenizer_2、transformer、vae，七个模型的配置文件及权重。
- device_id：推理设备ID。
- prompt：用于图像生成的文字描述提示。
- input_size：需要生成的图像尺寸。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数。
- use_lora：使用 --use_lora 开启LoRA风格化切换。
- lora_ckpt：LoRA权重路径。