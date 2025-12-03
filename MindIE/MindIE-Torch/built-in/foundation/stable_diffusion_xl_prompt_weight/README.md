# 模型推理指导  

## 一、模型简介

stable-diffusionxl-prompt-weighting描述增强(类似++的操作)：通过“提示权重（prompt weighting）”来精细调控模型对输入文本提示中不同概念的关注程度，从而影响最终生成图像的内容和焦点。
参考实现请查看[stable-diffusionxl-prompt-weighting blog](https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts#stable-diffusion-xl)。

## 二、环境准备

  **表 1**  版本配套表

  | 配套   | 版本    | 环境准备指导 |
  | ------ | ------- | ------------ |
  | Python | 3.10.13 | -            |
  | torch  | 2.1.0   | -            |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- 支持卡数：Atlas 800I A2支持的卡数为1；Atlas 300I Duo支持的卡数为1
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)
**注意**：该模型当前仅打通功能，性能受CPU规格影响，建议使用64核CPU（arm）。

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
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
```

### 2.5 安装所需依赖
按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
```shell
pip install -r requirements.txt
```

## 三、模型权重

下载权重，放到代码同级目录下，以避免执行后面步骤时可能会出现下载失败。
```bash
# 需要使用 git-lfs (https://git-lfs.com)
git lfs install

# xl
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

## 四、模型推理

### 4.1 代码修改
执行命令：
```bash
python3 stable_diffusion_clip_patch.py
python3 stable_diffusion_attention_patch.py

# 若使用unetCache
python3 stable_diffusionxl_unet_patch.py
```

### 4.2 模型转换
使用Pytorch导出pt模型，再使用MindIE推理引擎转换为适配昇腾的模型。

1. 设置权重路径：
```bash
# xl (执行时下载权重)
model_base="stabilityai/stable-diffusion-xl-base-1.0"

# xl (使用上一步下载的权重)
model_base="./stable-diffusion-xl-base-1.0"
```

2. 执行命令：
```bash
python3 export_ts_prompt_weight.py --model ${model_base} --output_dir ./models --batch_size 1 --flag 1 --soc A2 --device 0

```
参数说明：
- --model：模型权重路径
- --output_dir: ONNX模型输出目录
- --batch_size: 设置batch_size, 默认值为1,当前仅支持batch_size=1的场景
- --flag：默认为1。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512。
- --soc：只支持Duo和A2。默认为A2。
- --device：推理设备ID
- --use_cache: 【可选】在推理过程中使用cache

静态编译场景：

- ./models/clip/clip_bs{batch_size}.pt, ./models/clip/clip_bs{batch_size}_compile.ts 和 ./models/clip/clip2_bs{batch_size}.pt, ./models/clip/clip2_bs{batch_size}_compile.ts
- ./models/unet/unet_bs{batch_size x 2}.pt, ./models/unet/unet_bs{batch_size x 2}_compile_static.ts
- ./models/vae/vae_bs{batch_size}.pt, ./models/vae/vae_bs{batch_size}_compile_static.ts
- ./models/ddim/ddim_bs{batch_size}.pt, ./models/ddim/ddim_bs{batch_size}_compile_static.ts

动态分档场景：

- ./models/clip/clip_bs{batch_size}.pt, ./models/clip/clip_bs{batch_size}_compile.ts 和 ./models/clip/clip2_bs{batch_size}.pt, ./models/clip/clip2_bs{batch_size}_compile.ts
- ./models/unet/unet_bs{batch_size x 2}.pt, ./models/unet/unet_bs{batch_size x 2}_compile.ts
- ./models/vae/vae_bs{batch_size}.pt, ./models/vae/vae_bs{batch_size}_compile.ts
- ./models/ddim/ddim_bs{batch_size}.pt, ./models/ddim/ddim_bs{batch_size}_compile.ts
      
### 4.3 性能测试
执行推理脚本
```bash
# 不使用unetCache策略
python3 stable_diffusionxl_pipeline_prompt_weight.py \
         --model ${model_base} \
         --prompt_file ./prompts.txt \
         --device 0 \
         --save_dir ./results \
         --steps 50 \
         --output_dir ./models \
         --flag 1 \
         --w_h 1024 

# 使用UnetCache策略
python3 stable_diffusionxl_pipeline_prompt_weight.py \
         --model ${model_base} \
         --prompt_file ./prompts.txt \
         --device 0 \
         --save_dir ./results_unetCache \
         --steps 50 \
         --output_dir ./models \
         --flag 1 \
         --w_h 1024 \
         --use_cache
```

参数说明：
- --model: 模型名称或本地模型目录的路径。
- --prompt_file: 提示词文件。
- --device: 推理设备ID。
- --save_dir: 生成图片的存放目录。
- --steps: 生成图片迭代次数。
- --output_dir: 存放导出模型的目录。
- --flag：默认为1。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512。**注意**：请与导出模型时设置的flag保持一致
- --w_h: image的宽高，设置为1024表示宽高均为1024，设置为512表示宽高均为512。仅支持这两种分辨率。
- --use_cache: 【可选】在推理过程中使用cache。

执行完成后在 `./results`目录下生成推理图片。

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。