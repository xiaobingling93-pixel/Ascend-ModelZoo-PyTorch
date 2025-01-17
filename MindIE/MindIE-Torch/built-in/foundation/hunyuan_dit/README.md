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

### 1.5 安装mindspeed
```shell
# 下载mindspeed源码仓
git clone https://gitee.com/ascend/MindSpeed.git
# 使用pip安装
pip install -e MindSpeed
```

## 二、下载本仓库

### 2.1 下载到本地
```shell
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

### 2.2 安装依赖
使用pip安装
```shell
pip install -r requirents.txt
```
若要使用hpsv2验证精度，则还需要按照以下步骤安装hpsv2
```shell
git clone https://github.com/tgxs002/HPSv2.git
pip install -e HPSv2
```

## 三、HunyuanDiT使用

### 3.1 模型权重及配置文件说明
1. 权重链接:
```shell
https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2/tree/main/t2i
```
- 在t2i/model路径下，新增HunyuanDiT模型权重的配置文件，命名为config.json
```shell
{
  "_class_name": "HunyuanDiT2DModel",
  "_mindiesd_version": "2.0.RC1",
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
  "text_len_t5": 256,
  "size_cond": null,
  "use_style_cond": false
}
```
2. 各模型的配置文件、权重文件的路径层级样例如下所示。
```commandline
|----hunyuan_dit
|    |---- ckpts
|    |    |---- t2i
|    |    |    |---- clip_text_encoder
|    |    |    |---- model
|    |    |    |    |---- config.json
|    |    |    |    |---- 模型权重
|    |    |    |---- mt5
|    |    |    |---- sdxl-vae-fp16-fix
|    |    |    |---- tokenizer
```

### 3.2 模型单卡推理适配的测试
设置权重路径
```shell
path="ckpts/t2i"
```
修改权重文件夹权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
```
执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --prompt "渔舟唱晚" \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- prompt：用于图像生成的文字描述提示。
- input_size：需要生成的图像尺寸。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在"results"目录下生成推理图像，图像生成顺序与prompt顺序保持一致，并在终端显示推理时间。

### 3.3 模型单卡等价优化的性能/精度测试
设置权重路径
```shell
path="ckpts/hydit"
```
修改权重文件夹权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
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
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启prompt_list列表中的图像生成，用于性能/精度测试。
- prompt_list：用于图像生成的文字描述提示的列表文件路径。
- input_size：需要生成的图像尺寸。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在"results"目录下生成推理图像，图像生成顺序与prompt顺序保持一致，并在终端显示推理时间。

### 3.4 模型单卡算法优化的性能/精度测试
设置权重路径
```shell
path="ckpts/hydit"
```
修改权重文件夹权限为安全权限
```shell
chmod -R 640 ckpts/t2i/
```
执行命令：
```shell
python inference_hydit.py \
       --path ${path} \
       --device_id 0 \
       --test_acc \
       --prompt_list "prompts/example_prompts.txt" \
       --use_cache \
       --input_size 1024 1024 \
       --seed 42 \
       --infer_steps 100
```
参数说明：
- path：权重路径，包含clip_text_encoder、model、mt5、sdxl-vae-fp16-fix、tokenizer的权重及配置文件。
- device_id：推理设备ID。
- test_acc：使用 --test_acc 开启prompt_list列表中的图像生成，用于性能/精度测试。
- prompt_list：用于图像生成的文字描述提示的列表文件路径。
- use_cache：使用 --use_cache 开启算法策略优化的测试。
- input_size：需要生成的图像尺寸。
- seed：设置随机种子，默认值为42。
- infer_steps：推理迭代步数，默认值为100。

执行完成后在"results"目录下生成推理图像，图像生成顺序与prompt顺序保持一致，并在终端显示推理时间。

## 四、精度验证
由于生成的图片存在随机性，提供两种精度验证方法：
1. CLIP-score（文图匹配度量）：评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。使用Parti数据集进行验证。
2. HPSv2（图片美学度量）：评估生成图片的人类偏好评分，分数的取值范围为[0, 1]，越高越好。使用HPSv2数据集进行验证

【注意】由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

### 4.1 下载Parti数据集和hpsv2数据集
```shell
# 下载Parti数据集
wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
```
hpsv2数据集下载链接：https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_benchmark_prompts.json

### 4.2 下载模型权重
```shell
# Clip Score和HPSv2均需要使用的权重
GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
cd ./CLIP-ViT-H-14-laion2B-s32B-b79K
# HPSv2权重
wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt --no-check-certificate
```
也可手动下载[Clip Score权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)，将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下，手动下载[HPSv2权重](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)放到当前路径。