## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.12 | - |
  | torch | 2.4.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持：
Atlas 800I A2/Atlas 800T A2设备：支持的卡数为1
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
安装pytorch框架 版本2.4.0
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

## 三、CogView3使用

### 3.1 权重及配置文件说明
1. CogView3权重路径:
```shell
https://huggingface.co/THUDM/CogView3-Plus-3B/tree/main
```
- 修改该权重的model_index.json
```shell
{
  "_class_name": "CogView3PlusPipeline",
  "_diffusers_version": "0.31.0",
  "scheduler": [
    "cogview3plus",
    "CogVideoXDDIMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "T5EncoderModel"
  ],
  "tokenizer": [
    "transformers",
    "T5Tokenizer"
  ],
  "transformer": [
    "cogview3plus",
    "CogView3PlusTransformer2DModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```
2. scheduler权重链接:
```shell
https://huggingface.co/THUDM/CogView3-Plus-3B/tree/main/scheduler
```
3. text_encoder权重链接：
```shell
https://huggingface.co/THUDM/CogView3-Plus-3B/tree/main/text_encoder
```
4. tokenizer权重链接：
```shell
https://huggingface.co/THUDM/CogView3-Plus-3B/tree/main/tokenizer
```
5. transformer权重链接：
```shell
https://huggingface.co/THUDM/CogView3-Plus-3B/tree/main/transformer
```
6. vae权重链接：
```shell
https://huggingface.co/THUDM/CogView3-Plus-3B/tree/main/vae
```
7. 各模型的配置文件、权重文件的层级样例如下所示。
```commandline
|----CogView3B
|    |---- configuration.json
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
设置权重路径
```shell
model_path='/data/CogView3B'
```
执行命令：
```shell
python inference_cogview3plus.py \
       --model_path ${model_path} \
       --device_id 0 \
       --width 1024 \
       --height 1024 \
       --num_inference_steps 50 \
       --dtype bf16
```
参数说明：
- model_path：权重路径，包含scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重。
- device_id：推理设备ID。
- width：需要生成的图像的宽。
- height: 需要生成的图像的高。
- num_inference_steps：推理迭代步数。
- dtype: 数据类型。目前只支持bf16。

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。