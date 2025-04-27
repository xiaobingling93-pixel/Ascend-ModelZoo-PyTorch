# 模型推理指导  

## 一、模型简介

stable-video-diffusion是一种图像到视频的扩散模型，能够在给定图像输入的情况下生成与图像相对应的视频。参考实现请查看[Stable Video Diffusion blog](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)。

本模型使用的优化手段如下：
- 等价优化：FA、DP并行

本模型输入输出数据：
  **表 1**  输入数据

  | 输入数据  | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | input    |  1 x 512 x 512 x 3 | FLOAT32 |  NHWC |

  **表 2**  输出数据

  | 输出数据 | 大小      | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output   | 1 x 25 x 512 x 512 x 3 | FLOAT32  | NTHWC |

  **注意**：该模型当前仅支持batch size为1的情况。

## 二、环境准备

  **表 3**  版本配套表

  | 配套                                                         | 版本     | 备注                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
  | torch | 2.0.0  | 导出pt模型所需版本                                            |
  | torch | 2.1.0  | 模型编译和推理所需版本                                         |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- 支持卡数：支持的卡数为1或2
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

### 2.4 下载本仓库
```shell
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```

### 2.5 安装所需依赖
按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
```shell
pip install -r requirements.txt
```

## 三、模型权重

### 3.1 权重下载
```bash
# 需要使用 git-lfs (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
```

### 3.2 获取原始数据集
本模型输入图像示例的下载网址为：https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png
用户自网址自行下载后放置当前路径下，命名为 rocket.png

## 四、模型推理

### 4.1 代码修改
执行命令：
```bash
python3 stable_video_diffusion_activations_patch.py
python3 stable_video_diffusion_attention_patch.py
python3 stable_video_diffusion_transformer_patch.py
```

### 4.2 模型转换
使用Pytorch导出pt模型，再使用MindIE推理引擎转换为适配昇腾的模型。

1. 设置权重路径：
```bash
# 执行时下载权重
model_base="stabilityai/stable-video-diffusion-img2vid-xt"

# 使用上一步下载的权重
model_base="./stable-video-diffusion-img2vid-xt"
```

2. 执行命令：
```bash
# 导出pt模型
python3 export_ts.py --model ${model_base} --output_dir ./models
# 更换torch版本，执行后续的模型编译和推理
python3 uninstall torch
python3 install torch==2.1.0
```

参数说明：
- --model：模型名称或本地模型目录的路径
- --output_dir: pt模型输出目录

执行成功后会生成pt模型:
   - ./models/image_encoder_embed/image_encoder_embed.pt
   - ./models/unet/unet_bs2.pt
   - ./models/vae/vae_encode.pt
   - ./models/vae/vae_decode.pt

### 4.3 性能测试
1. 开启cpu高性能模式：
```bash
echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
```

2. 安装绑核工具：
```bash
apt-get update
apt-get install numactl
```
查询卡的NUMA node
```shell
lspci -vs bus-id
```
bus-id可通过npu-smi info获得，查询到NUMA node，在推理命令前加上对应的数字

可通过lscpu获得NUMA node对应的CPU核数
```shell
NUMA node0: 0-23
NUMA node1: 24-47
NUMA node2: 48-71
NUMA node3: 72-95
```
当前查到NUMA node是0，对应0-23，推荐绑定其中单核以获得更好的性能。

3. 执行命令：
```bash
# 0.第一次推理需要配置环境变量，使得在静态TLS块中可以分配内存：
find / -name *libGL* # 查找libGLdispatch.so.0文件的路径，记为lib_dir，例如 lib_dir="/lib/aarch64-linux-gnu"
export LD_PRELOAD=${lib_dir}/libGLdispatch.so.0:$LD_PRELOAD

# 1.若不使用并行推理：
numactl -C 0-23 python3 stable_video_diffusion_pipeline.py \
         --model ${model_base} \
         --img_file ./rocket.png \
         --device 0 \
         --save_dir ./results \
         --num_inference_steps 25 \
         --output_dir ./models

# 2.若使用并行推理：
numactl -C 0-23 python3 stable_video_diffusion_pipeline_parallel.py \
         --model ${model_base} \
         --img_file ./rocket.png \
         --device 0,1 \
         --save_dir ./results \
         --num_inference_steps 25 \
         --output_dir ./models
```

参数说明：
- --model：模型名称或本地模型目录的路径。
- --img_file：输入图像文件。
- --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
- --save_dir：生成视频的存放目录。
- --num_inference_steps：生成视频的迭代次数。
- --output_dir: 编译好的模型路径。

执行完成后在`./results`目录下生成推理视频。并在终端显示推理时间。

**注意**：若使用Atlas 800I A2单卡推理，则需要保证单卡的实际可用内存（最大值-无进程时初始值）> 29762MB。否则尝试重启服务器以降低无进程时初始值、更换服务器，或使用双卡并行推理。
**注意**：当前推理pipline中未固定随机种子，固定方式如下：
```bash
# 推理pipline main函数中加入
generator = torch.Generator().manual_seed(xxx)
# 在ascendie_infer函数中加入参数
generator=generator
```

## 五、推理结果参考
### StableVideoDiffusion性能数据
| 硬件形态 | batch size | 迭代次数 | 数据类型 | 卡数 | 性能 |
| :------: |:----:|:----:|:----:|:----:|:----:|
| Atlas 800I A2(8*32G) |  1  |  25  | float16 | 1 |  28s    |
| Atlas 800I A2(8*32G) |  1  |  25  | float16 | 2 |  14.5s  |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。