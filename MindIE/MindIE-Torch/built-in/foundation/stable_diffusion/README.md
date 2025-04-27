# 模型推理指导  

## 一、模型简介

stable-diffusion是一种文本到图像的扩散模型，能够在给定任何文本输入的情况下生成照片逼真的图像。有关稳定扩散函数的更多信息，请查看[Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion)。

本模型使用的优化手段如下：
- 等价优化：FA、DP并行
- 算法优化：UnetCache

本模型输入输出数据：
  **表 1**  输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    |  1 x 77 | FLOAT32|  ND|

  **表 2**  输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 512 x 512 x 3 | FLOAT32  | NHWD           |

**注意**：该模型当前仅支持batch size为1的情况。

## 二、环境准备

  **表 3**  版本配套表

  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
  | torch| 2.1.0  | -                                                            |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- 支持卡数：Atlas 800I A2支持的卡数为1或2；Atlas 300I Duo支持的卡数为1，可双芯并行
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

**注意**：本README中的StableDiffusion v1.5和v2.1模型推理方式与torch-npu冲突，需卸载torch-npu包。

## 三、模型权重

权重下载
```bash
# 需要使用 git-lfs (https://git-lfs.com)
git lfs install

# v1.5
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

# v2.1
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
```

## 四、模型推理

### 4.1 代码修改
执行命令：
```bash
python3 stable_diffusion_attention_patch.py
python3 stable_diffusion_unet_patch.py
```

### 4.2 模型转换【可选】
使用Pytorch导出pt模型，再使用MindIE推理引擎转换为适配昇腾的模型。

1. 设置权重路径：
```bash
# v1.5 (执行时下载权重)
model_base="stable-diffusion-v1-5/stable-diffusion-v1-5"

# v1.5 (使用上一步下载的权重)
model_base="./stable-diffusion-v1-5"

# v2.1 (执行时下载权重)
model_base="stabilityai/stable-diffusion-2-1-base"

# v2.1 (使用上一步下载的权重)
model_base="./stable-diffusion-2-1-base"
```

2. 执行命令：
```bash
# 导出pt模型
python3 export_ts.py --model ${model_base} --output_dir ./models \
         --parallel \
         --use_cache
```

参数说明：
- --model：模型名称或本地模型目录的路径
- --output_dir: pt模型输出目录
- --parallel：【可选】模型使用双芯/双卡并行推理
- --use_cache: 【可选】模型使用UnetCache优化
- --use_cache_faster: 【可选】模型使用deepcache+faster融合方案

若不选择【--parallel】，即单卡/单芯，执行成功后会生成pt模型:
   - ./models/clip/clip_bs1.pt
   - ./models/vae/vae_bs1.pt
   - ./models/ddim/ddim2.pt
   - ./models/cat/cat.pt
   - ./models/unet/unet_bs2.pt【不选择--use_cache】
   - ./models/unet/unet_bs2_0.pt【选择--use_cache】
   - ./models/unet/unet_bs2_1.pt【选择--use_cache】

若选择【--parallel】，即双卡/双芯，执行成功后会生成pt模型:
   - ./models/clip/clip_bs1.pt
   - ./models/vae/vae_bs1.pt
   - ./models/ddim/ddim1.pt
   - ./models/unet/unet_bs1.pt【不选择--use_cache】
   - ./models/unet/unet_bs1_0.pt【选择--use_cache】
   - ./models/unet/unet_bs1_1.pt【选择--use_cache】

**注意**：若条件允许，该模型可以双芯片并行的方式进行推理，从而获得更短的端到端耗时。具体指令的差异之处会在后面的步骤中单独说明，请留意。

### 4.3 使用Lora权重进行模型转换【可选】
1. 权重转换：
在[civitai](https://civitai.com)下载base model为SD1.5和SD2.1的的lora权重，一般选择safetensor格式的权重。执行转换脚本，将lora权重和model_base权重结合在一起。
```bash
model_lora=lora权重路径
model_new=适配lora之后的SD权重路径
python3 convert_lora_safetensors_to_diffusers.py --base_model_path ${model_base} --checkpoint_path ${model_lora} --dump_path ${model_new}
```

2. 执行命令：
```bash
# 使用lora权重，导出pt模型
python3 export_ts.py --model ${model_new} --output_dir ./models_lora \
         --parallel \
         --use_cache
```

执行成功后会在./models_lora路径下生成pt模型：

**注意**：更换lora权重时，请手动删除models_lora路径的生成的pt模型，重新执行转换权重脚本和导出模型命令导出带lora权重的pt模型。

### 4.4 性能测试
1. 开启cpu高性能模式：
```bash
echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
```

2. 安装绑核工具
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
# 1.若不使用并行推理：
# 1.1不使用lora权重
numactl -C 0-23 python3 stable_diffusion_pipeline.py \
         --model ${model_base} \
         --prompt_file ./prompts.txt \
         --device 0 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc A2 \
         --output_dir ./models \
         --use_cache

# 1.2使用带lora权重的新权重
numactl -C 0-23 python3 stable_diffusion_pipeline.py \
         --model ${model_new} \
         --prompt_file ./prompts.txt \
         --device 0 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc A2 \
         --output_dir ./models_lora \
         --use_cache

# 2.若使用并行推理【Atlas 300I Duo】
# 2.1不使用lora权重
numactl -C 0-23 python3 stable_diffusion_pipeline_parallel.py \
         --model ${model_base} \
         --prompt_file ./prompts.txt \
         --device 0,1 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc Duo \
         --output_dir ./models \
         --use_cache

# 2.2使用带lora权重的新权重
numactl -C 0-23 python3 stable_diffusion_pipeline_parallel.py \
         --model ${model_new} \
         --prompt_file ./prompts.txt \
         --device 0,1 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc Duo \
         --output_dir ./models_lora \
         --use_cache
```

参数说明：
- --model：模型名称或本地模型目录的路径。
- --prompt_file：输入文本文件，按行分割。
- --save_dir：生成图片的存放目录。
- --steps：生成图片迭代次数。
- --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
- --scheduler: 【可选】推荐使用DDIM采样器。
- --soc: 硬件配置，根据硬件配置选择Duo或者A2。
- --output_dir: 编译好的模型路径。
- --use_cache: 【可选】推荐使用UnetCache策略。
- --use_cache_faster: 【可选】模型使用deepcache+faster融合方案。

执行完成后会在`./results`目录下生成推理图片。并在终端显示推理时间。

**注意**：更换lora权重时，请手动删除models_lora路径的生成的编译好的pt模型，（xxx_compile.pt）重新执行推理脚本。
**注意**：当前推理pipline中未固定随机种子，固定随机种子会对clip_score分数有影响，固定方式如下：
```bash
# 推理pipline main函数中加入
generator = torch.Generator().manual_seed(xxx)
# 在ascendie_infer函数中加入参数
generator=generator
```

### 4.5 精度测试
由于生成的图片存在随机性，所以精度验证将使用CLIP-score来评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。

**注意**：由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

1. 下载Parti数据集

```bash
wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
```

2. 下载Clip模型权重

```bash
# 安装git-lfs
apt install git-lfs
git lfs install
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

# 或者访问https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin，将权重下载并放到这个目录下
```

2. 使用推理脚本读取Parti数据集，生成图片
```bash
# 1.若不使用并行推理：
# 1.1不使用lora权重
python3 stable_diffusion_pipeline.py \
         --model ${model_base} \
         --prompt_file ./PartiPrompts.tsv \
         --prompt_file_type parti \
         --num_images_per_prompt 4 \
         --device 0 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc A2 \
         --output_dir ./models \
         --use_cache

# 1.2使用带lora权重的新权重
python3 stable_diffusion_pipeline.py \
         --model ${model_new} \
         --prompt_file ./PartiPrompts.tsv \
         --prompt_file_type parti \
         --num_images_per_prompt 4 \
         --device 0 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc A2 \
         --output_dir ./models_lora \
         --use_cache

# 2.若使用并行推理【Atlas 300I Duo】
# 2.1不使用lora权重
python3 stable_diffusion_pipeline_parallel.py \
         --model ${model_base} \
         --prompt_file ./PartiPrompts.tsv \
         --prompt_file_type parti \
         --num_images_per_prompt 4 \
         --device 0,1 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc Duo \
         --output_dir ./models \
         --use_cache

# 2.2使用带lora权重的新权重
python3 stable_diffusion_pipeline_parallel.py \
         --model ${model_new} \
         --prompt_file ./PartiPrompts.tsv \
         --prompt_file_type parti \
         --num_images_per_prompt 4 \
         --device 0,1 \
         --save_dir ./results \
         --steps 50 \
         --scheduler DDIM \
         --soc Duo \
         --output_dir ./models_lora \
         --use_cache
```

增加的参数说明：
- --prompt_file：输入文本文件，按行分割。
- --prompt_file_type: prompt文件类型，用于指定读取方式。
- --num_images_per_prompt: 每个prompt生成的图片数量。

执行完成后会在`./results`目录下生成推理图片，并且会在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。

4. 计算CLIP-score

```bash
python clip_score.py \
         --device=cpu \
         --image_info="image_info.json" \
         --model_name="ViT-H-14" \
         --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
```

参数说明：
- --device: 推理设备。
- --image_info: 上一步生成的`image_info.json`文件。
- --model_name: Clip模型名称。
- --model_weights_path: Clip模型权重文件路径。

执行完成后会在屏幕打印出精度计算结果。


## 五、推理结果参考
### StableDiffusion v1.5 性能 & 精度数据

| 硬件形态 | batch size | 迭代次数 | 数据类型 | 性能(w/o UnetCache) | 性能(with UnetCache) | CLIP_score |
| :------: |:----:|:----:|:----:|:----:|:----:|:----:|
| Atlas 300I Duo双芯 |  1  |  50  | float16 | 2.5s | 1.54s | 0.363 |
| Atlas 800I A2(8*32G) |  1  |  50  | float16 |  1.6s  |  0.95s  | 0.363 |

### StableDiffusion v2.1 性能 & 精度数据

| 硬件形态 | batch size | 迭代次数 | 数据类型 | 性能(w/o UnetCache) | 性能(with UnetCache) | CLIP_score |
| :------: |:----:|:----:|:----:|:----:|:----:|:----:|
| Atlas 300I Duo双芯 |  1  |  50  | float16 | 2.3s | 1.39s | 0.376 |
| Atlas 800I A2(8*32G) |  1  |  50  | float16 |  1.4s  |  0.85s  | 0.376 |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。