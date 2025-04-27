---
license: apache-2.0
frameworks:
  - PyTorch
language:
  - en
hardwares:
  - NPU
---

# 模型推理指导

## 一、模型简介

CogView是一种文本到图像的扩散模型，能够在给定文本输入的情况下生成相符的图像。

本模型使用的优化手段如下：
- 等价优化：FA、Linear
- 算法优化：FA、Linear、cache

## 二、环境准备

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10 / 3.11 | - |
  | torch | 2.1.0 | - |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 800T A2](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26)
- 支持卡数：支持的卡数为1
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

### 2.5 下载本仓库
```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

### 2.6 安装所需依赖
```shell
pip install -r requirements.txt
```

## 三、模型权重

### 3.1 权重下载
下载CogView权重:
```shell
   git clone https://huggingface.co/THUDM/CogView3-Plus-3B/tree/main
```

### 3.2 配置文件说明
```commandline
|----main
|    |---- configuration.json
|    |---- model_index.json
|    |---- scheduler
|    |---- text_encoder
|    |---- tokenizer
|    |---- transformer
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- vae
```

## 四、模型推理

### 4.1 单卡单batch性能测试
1. 设置权重路径：
```shell
path="/data/CogView3B"
```

2. 执行命令：
```shell
python inference_cogview3plus.py \
       --model_path ${path} \
       --prompt_file ./prompts/example_prompts.txt \
       --width 1024 \
       --height 1024 \
       --num_inference_steps 50 \
       --dtype bf16 \
       --device_id 0 \
       --cache_algorithm attention
```
参数说明：
- model_path：权重路径，包含scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重。
- prompt_file：提示词文件。
- width：需要生成的图像的宽。
- height: 需要生成的图像的高。
- num_inference_steps：推理迭代步数。
- dtype: 数据类型。目前只支持bf16。
- device_id：推理设备ID。
- cache_algorithm：默认为None，可选择attention，即使用AGBCache算法，注意是有损的加速算法。

**注意**：在32G的服务器上，开启cache算法可能会报显存不足的错误；在64G机器上，可正常开启cache算法。
**注意**：本仓库模型，是对开源模型进行优化。用户在使用时，应对开源代码函数的变量范围，类型进行校验，避免出现变量超出范围、除零等操作。


### 4.2 单卡多batch性能测试
1. 设置权重路径
```shell
path="/data/CogView3B"
```

2. 执行命令：
```shell
python inference_cogview3plus.py \
       --model_path ${path} \
       --prompt_file ./prompts/example_prompts.txt \
       --width 1024 \
       --height 1024 \
       --num_inference_steps 50 \
       --dtype bf16 \
       --batch_size 2 \
       --device_id 0 \
       --cache_algorithm attention
```
参数说明：
- model_path：权重路径，包含scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重。
- prompt_file：提示词文件。
- width：需要生成的图像的宽。
- height: 需要生成的图像的高。
- num_inference_steps：推理迭代步数。
- dtype: 数据类型。目前只支持bf16。
- batch_size: 推理时的batch_size。
- device_id：推理设备ID。
- cache_algorithm：默认为None，可选择attention，即使用AGBCache算法，注意是有损的加速算法。

**注意**：在32G的服务器上，batch_size需要等于1，否则会报显存不足的错误；在64G机器上，batch_size可为2，可开启cache算法。


### 4.3 精度测试
由于生成的图片存在随机性，提供两种精度验证方法：
- CLIP-score（文图匹配度量）：评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。使用Parti数据集进行验证。
- HPSv2（图片美学度量）：评估生成图片的人类偏好评分，分数的取值范围为[0, 1]，越高越好。使用HPSv2数据集进行验证

**注意**：由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

##### 1. 读取数据集生成图像
1. 下载Parti数据集和hpsv2数据集
所有数据集放到`cogview3/prompts`目录下
```bash
# 下载Parti数据集
wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
```
hpsv2数据集下载链接：https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_benchmark_prompts.json

2. 设置模型权重路径
```shell
path="/data/CogView3B"
```

3. 使用推理脚本读取Parti数据集，生成图片
```bash
python3 inference_cogview3plus.py \
        --model_path ${path} \
        --prompt_file ./prompts/PartiPrompts.tsv \
        --prompt_file_type parti \
        --info_file_save_path ./image_info_PartiPrompts.json \
        --save_dir ./results_PartiPrompts \
        --num_images_per_prompt 4 \
        --height 1024 \
        --width 1024 \
        --batch_size 1 \
        --seed 42 \
        --device_id 0 \
       --cache_algorithm attention
```
参数说明：
- model_path：权重路径，包含scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重。
- prompt_file：提示词文件。
- prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。
- info_file_save_path：生成图片信息的json文件路径。
- save_dir：生成图片的存放目录。
- num_images_per_prompt: 每个prompt生成的图片数量。注意使用hpsv2时，设置num_images_per_prompt=1即可。
- height: 需要生成的图像的高。
- width：需要生成的图像的宽。
- batch_size：模型batch size。
- seed：随机种子。
- device_id：推理设备ID。
- cache_algorithm：默认为None，可选择attention，即使用AGBCache算法，注意是有损的加速算法。

执行完成后在`./results_PartiPrompts`目录下生成推理图片，在当前目录生成一个`image_info_PartiPrompts.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。

4. 使用推理脚本读取hpsv2数据集，生成图片
```bash
python3 inference_cogview3plus.py \
        --model_path ${path} \
        --prompt_file ./prompts/hpsv2_benchmark_prompts.json \
        --prompt_file_type hpsv2 \
        --info_file_save_path ./image_info_hpsv2.json \
        --save_dir ./results_hpsv2 \
        --num_images_per_prompt 1 \
        --height 1024 \
        --width 1024 \
        --batch_size 1 \
        --seed 42 \
        --device_id 0 \
       --cache_algorithm attention
```
参数说明：
- model_path：权重路径，包含scheduler、text_encoder、tokenizer、transformer、vae，5个模型的配置文件及权重。
- prompt_file：提示词文件。
- prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。
- info_file_save_path：生成图片信息的json文件路径。
- save_dir：生成图片的存放目录。
- num_images_per_prompt: 每个prompt生成的图片数量。注意使用hpsv2时，设置num_images_per_prompt=1即可。
- height: 需要生成的图像的高。
- width：需要生成的图像的宽。
- batch_size：模型batch size。
- seed：随机种子。
- device_id：推理设备ID。
- cache_algorithm：默认为None，可选择attention，即使用AGBCache算法，注意是有损的加速算法。

执行完成后在`./results_hpsv2`目录下生成推理图片，在当前目录生成一个`image_info_hpsv2.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。

##### 2. 计算精度指标(GPU)
1. 下载模型权重
所有权重下载到`cogview3/`目录下
```bash
# Clip Score和HPSv2均需要使用的权重
# 安装git-lfs
apt install git-lfs
git lfs install

# Clip Score权重
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

# HPSv2权重
wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt --no-check-certificate
```
也可手动下载[CLIP权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下，手动下载[HPSv2权重](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)放到当前路径

2. CLIP-score精度指标计算
```bash
python3 clip_score.py \
      --device=cuda \
      --image_info="./image_info_PartiPrompts.json" \
      --model_name="ViT-H-14" \
      --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
```
参数说明：
- --device: 推理设备（CPU或者GPU）。
- --image_info: 上一步生成的`image_info_PartiPrompts.json`文件。
- --model_name: Clip模型名称。
- --model_weights_path: Clip模型权重文件路径。

执行完成后会在屏幕打印出精度计算结果。

3. HPSv2精度指标计算
```bash
python3 hpsv2_score.py \
      --image_info="image_info_hpsv2.json" \
      --HPSv2_checkpoint="./HPS_v2_compressed.pt" \
      --clip_checkpoint="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
```

参数说明：
- --image_info: 上一步生成的`image_info_hpsv2.json`文件。
- --HPSv2_checkpoint: HPSv2模型权重文件路径。
- --clip_checkpointh: Clip模型权重文件路径。

执行完成后会在屏幕打印出精度计算结果。

## 五、推理结果参考
### CogView3plus性能 & 精度数据
| 硬件形态 | 迭代次数 | 加速算法 | 性能 | CLIP_score | HPSV2_score |
| :------: |:----:|:----:|:----:|:----:|:----:|
| Atlas 800T A2 (8*64G) 单卡 |  50  |  无  |  27.588s  |  0.367  |  0.2879729  |
| Atlas 800T A2 (8*64G) 单卡 |  50  |  AGBCache   | 17.219s  |  0.367  |  0.2879835  |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。