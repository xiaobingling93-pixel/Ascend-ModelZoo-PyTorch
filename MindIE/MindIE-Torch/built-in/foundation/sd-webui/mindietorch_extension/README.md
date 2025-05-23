# 模型推理指导  

## 一、模型简介
SDWebUI是一个基于Gradio库的WebUi界面，支持设置输入和参数用于SD模型的文生图、图生图等功能。有关SDWebUI的更多信息，请查看[Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)。
mindietorch_extension实现了一个SDWebUI界面的插件，用优化后的diffusers.Unet2DConditionModel替换原有的UNetModel进行推理，支持SD文生图和图生图功能。底层调用了MindIE的build编译优化功能，通过PASS改图、Batch并行等优化手段，提升了推理性能。

## 二、环境准备

  **表 1**  版本配套表

  | 配套                                 | 版本    | 环境准备指导 |
  | ------------------------------------ | ------- | ------------ |
  | Python                               | 3.10    | -            |
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

### 2.4 获取源码 & 本仓库代码
1. 拉取webui工程代码[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
mkdir repositories && cd repositories
git clone https://github.com/Stability-AI/stablediffusion stable-diffusion-stability-ai
git clone https://github.com/Stability-AI/generative-models.git
git clone https://github.com/crowsonkb/k-diffusion.git
git clone https://github.com/sczhou/CodeFormer.git
git clone https://github.com/salesforce/BLIP.git
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets
```

2. 下载clip-vit-large-patch14，放在自定义路径下
```bash
git lfs install
git clone https://huggingface.co/openai/clip-vit-large-patch14
```

修改webui的源码：

①文件：stable-diffusion-webui/repositories/stable-diffusion-stability-ai/ldm/modules/encoders/modules.py

将该文件中涉及到的version="open/clip-vit-large-patch14"改为vesion=“下载的clip-vit-large-patch14路径”

②文件：stable-diffusion-webui/repositories/generative-models/sgm/modules/encoders/modules.py

将该文件中涉及到的version="open/clip-vit-large-patch14"改为vesion=“下载的clip-vit-large-patch14路径”

3. 拉取本仓库mindietorch_extension工程，放在stable-diffusion-webui/extensions路径下

### 2.5 安装所需依赖
按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
```
pip install -r requirements.txt
```

## 三、模型权重
1. 获取权重
```bash
# 需要使用 git-lfs (https://git-lfs.com)
git lfs install

# v1.5，将该权重放在stable-diffusion-webui/extensions/mindietorch_extension/models路径下
cd stable-diffusion-webui/extensions/torch_aie_extension/models
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5

# v2.1，将该权重放在stable-diffusion-webui/extensions/mindietorch_extension/models路径下
git clone https://huggingface.co/runwayml/stable-diffusion-2-1-base

# sdxl，将该权重放在stable-diffusion-webui/extensions/mindietorch_extension/models路径下
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

2. 将特定权重放在stable-diffusion-webui/models/Stable-diffusion路径下。注意：本插件支持的webui权重如下：
```bash
# v1.5 二选一即可，推荐safetensors
v1-5-pruned-emaonly.safetensors
v1-5-pruned-emaonly.ckpt
# v2.1 二选一即可，推荐safetensors
v2-1_512_ema-pruned.safetensors
v2-1_512_ema-pruned.ckpt
# SDXL
sd_xl_base_1.0.safetensors
```

```bash
# 举例：
cp stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors ../../../models/Stable-diffusion
```

## 四、模型推理

### 4.1 代码修改
```bash
cd /……/stable-diffusion-webui/extensions/mindietorch_extension
# 修改attention，用于trace正确的模型
python sd_webui_patch.py

# 将mindietorch_extension工程的diff1.patch放到stable-diffusion-webui路径下
mv diff_1.patch ../..
patch -p0 < diff_1.patch
```

### 4.2 执行命令启动webui
```bash
cd /……/stable-diffusion-webui
python launch.py --skip-torch-cuda-test --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half --skip-prepare-environment
```

界面启动后，请先选择硬件配置，Duo或A2。然后选择MindIE_torch按钮，第一次启动服务时，点击MindIE_torch按钮后，会对于原始模型做一些处理，请耐心等待，直到服务端显示"You can generate image now!"字样后，再根据上述参数配置，点击generate生成结果。

**注意**：使用该插件后，原始的webui界面中的某些配置受到限制，可配置参数：
```
Sampling method
Sampling steps
CFG Scale
Seed
```

受限制参数：
```
使用SD1.5和SD2.1时，Width和Height都要设置为512
使用SDXL时，Width和Height都要设置为1024
Batch count要固定为1
Batch size要固定为1
```


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。