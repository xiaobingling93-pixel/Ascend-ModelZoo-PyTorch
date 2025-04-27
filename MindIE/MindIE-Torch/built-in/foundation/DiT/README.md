# 模型推理指导  

## 一、模型简介

DiT是一种基于Transformer的扩散模型，全称为Diffusion Transformer，DiT遵循ViT的技术方法。有关DiT模型的更多信息，请参考[DiT github](https://github.com/facebookresearch/DiT)。

本模型输入输出数据：
  **表 1**  输入数据

  | 输入数据 | 数据类型 | 大小                                       | 数据排布格式 |
  | -------- | -------- | ------------------------------------------ | ------------ |
  | x        | FLOAT32  | batch_size x 4 x latent_size x latent_size | NCHW         |
  | t        | INT64    | batch_size                                 | ND           |
  | y        | INT64    | batch_size                                 | ND           |

- image_num：需要生成的图片数量
- batch_size：image_num * 2
- latent_size：image_size // 8

  **表 2**  输出数据

  | 输出数据 | 数据类型 | 大小                                      | 数据排布格式 |
  | -------- | -------- | ----------------------------------------- | ------------ |
  | output   | FLOAT32  | image_num x 4 x latent_size x latent_size | NCHW         |

## 二、环境准备

  **表 3**  版本配套表

  | 配套                                 | 版本    | 环境准备指导 |
  | ------------------------------------ | ------- | ------------ |
  | Python                               | 3.10.13 | -            |
  | PyTorch                              | 2.1.0   | -            |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- 支持卡数：Atlas 800I A2支持的卡数为1；Atlas 300I Duo支持的卡数为1
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

### 2.4 获取源码
获取源码，并将当前目录下的如下文件移到DiT工程下
```bash
git clone https://github.com/facebookresearch/DiT
mv background_runtime.py export_model.py models_npu.py sample_npu.py vision.patch timm_patch.py requirements.txt fid_test.py ./DiT
```

### 2.5 安装所需依赖
```shell
pip install -r requirements.txt
```

## 三、模型权重

DiT权重文件下载链接如下，按需下载：

[DiT-XL-2-256x256下载链接](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)

[DiT-XL-2-512x512下载链接](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt)

vae权重文件下载链接如下，按需下载：

```bash
# ema
git clone https://huggingface.co/stabilityai/sd-vae-ft-ema
# mse
git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
```

## 四、模型推理

### 4.1 代码修改
执行命令：
```bash
cd ./DiT
# 若环境没有patch工具，请自行安装
python3 timm_patch.py
```

### 4.2 模型转换
该步骤会生成编译之后的pt模型
```bash
# Atlas 300I Duo卡
python3 export_model.py \
   --ckpt ./DiT-XL-2-512x512.pt \
   --vae_model ./sd-vae-ft-mse \
   --image_size 512 \
   --device 0 \
   --soc Duo \
   --output_dir ./models \
   --parallel

# Atlas 800I A2
python3 export_model.py \
   --ckpt ./DiT-XL-2-512x512.pt \
   --vae_model ./sd-vae-ft-mse \
   --image_size 512 \
   --device 0 \
   --soc A2 \
   --output_dir ./models
```

参数说明：

- --ckpt：DiT-XL-2的权重路径
- --vae_model：vae的权重路径
- --image_size：分辨率，支持256和512。默认为512
- --device：使用哪张卡
- --soc：soc_version，只支持Duo和A2
- --output_dir：pt模型输出目录
- --parallel：【可选】模型使用并行进行推理

### 4.3 性能测试

1. 开启cpu高性能模式
```bash
echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
```

2.  执行命令：
```bash
# Atlas 300I Duo
python3 sample_npu.py \
   --vae mse \
   --image_size 512 \
   --ckpt ./DiT-XL-2-512x512.pt \
   --device 0 \
   --class_label 0 \
   --output_dir ./models \
   --parallel

# Atlas 800I A2
python3 sample_npu.py \
   --vae mse \
   --image_size 512 \
   --ckpt ./DiT-XL-2-512x512.pt \
   --device 0 \
   --class_label 0 \
   --output_dir ./models \
   --warmup
```

参数说明：

- --vae：使用哪种vae模型，支持mse和ema
- --image_size：分辨率，支持256和512。默认为512
- --ckpt：DiT-XL-2的权重路径
- --device：使用哪张卡
- --class_label：可在0~999中任意指定一个整数，代表image_net的种类
- --output_dir：上一步骤指定的pt模型输出目录
- --parallel：【可选】模型使用并行进行推理
- --warmup:【可选】使用warmup可使得时间更准确。并行场景使用该选项会有问题，不建议使用

执行完成后会在当前路径生成sample.png

### 4.4 精度测试
1. 下载数据集
[ImageNet512x512](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz)(VIRTUAL_imagenet512.npz)和[ImageNet256x256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)(VIRTUAL_imagenet256_labeled.npz)

2. 使用脚本读取数据集，生成图片
```bash
# Atlas 300I Duo
python3 fid_test.py \
   --vae mse \
   --image_size 512 \
   --ckpt ./DiT-XL-2-512x512.pt \
   --device 0 \
   --output_dir ./models \
   --parallel \
   --results results

# Atlas 800I A2
   python3 fid_test.py \
   --vae mse \
   --image_size 512 \
   --ckpt ./DiT-XL-2-512x512.pt \
   --device 0 \
   --output_dir ./models \
   --results results
```

参数说明：

- --results：生成的1000张图片存放路径
- image_size：分辨率，支持256和512。默认为512

3. 计算FID：
```bash
# 512分辨率使用VIRTUAL_imagenet512.npz数据集
python3 -m pytorch_fid ./VIRTUAL_imagenet512.npz ./results
# 256分辨率使用VIRTUAL_imagenet256_labeled.npz数据集
python3 -m pytorch_fid ./VIRTUAL_imagenet256_labeled.npz ./results 
```

## 五、推理结果参考
### 模型推理性能

| 分辨率 | 硬件形态 | 迭代次数 | 数据类型 | 性能 |
| ------ | -------- | -------- | -------- | -------- |
| 512    | Atlas 300I Duo      | 250      | float16 | 19.6s    |
|        | Atlas 800I A2 (32G) | 250      | float16 | 10.49s   |
| 256    | Atlas 300I Duo      | 250      | float16 | 9.5s     |
|        | Atlas 800I A2 (32G) | 50       | float16 | 4.13s    |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。