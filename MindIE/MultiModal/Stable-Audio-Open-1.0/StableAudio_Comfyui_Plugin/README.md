# Stable-Audio-Open-1.0 ComfyUI推理指导
## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I A2推理设备：支持的卡数为1
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

## 二、下载ComfyUI
ComfyUI是一个基于节点的stable diffusion用户界面，用户能够通过链接不同的节点来构建复杂的图像生成工作流程，图像生成速度上超越传统WebUI，而且显存占用更少。
### 2.1 下载ComfyUI官方代码仓，并安装相关依赖
```shell
git clone https://github.com/comfyanonymous/ComfyUI
```
### 2.2 下载相关依赖
```shell
cd ComfyUI
pip install -r requirements.txt
```

## 三、加载MindIE SD 加速插件

### 3.1 下载插件并导入ComfyUI
```shell
git clone https://modelers.cn/MindIE/stable_audio_open_1.0.git
cp -r stable_audio_open_1.0/StableAudio_Comfyui_Plugin /PATH/TO/ComfyUI/custom_nodes/
```
注：`/PATH/TO/ComfyUI`为ComfyUI下载路径

### 3.2 依赖安装
```bash
cd /PATH/TO/ComfyUI/custom_nodes/StableAudio_Comfyui_Plugin/
pip3 install -r requirements.txt
apt-get update
apt-get install libsndfile1
```
### 3.3 权重下载 
下载text_encoder权重，保存为`t5_base.safetensors`，并将其放在 `ComfyUI/models/text_encoders/`路径下
```bash
https://huggingface.co/google-t5/t5-base/blob/main/model.safetensors 
```
下载diffusion_model权重，保存为`stable_audio_open_1.0.safetensors`，并将其放在`ComfyUI/models/checkpoints/`路径下
```bash
https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main ，
```

下载transformers权重，新建文件夹`ComfyUI/models/mindiesd/stable-audio-open-1.0/transformer`，并保存在该路径下(包含`config.json`)
```bash
https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main/transformer
```

## 四、启动ComfyUI
### 4.1 执行ComfyUI启动脚本
```shell
python main.py --gpu-only --listen
```

参数说明：
- --gpu-only：只在device（gpu/npu）上执行推理
- --listen：服务器监听的的ip地址，默认0.0.0.0

执行成功后会显示
```shell
To seethe GUI go to: http://0.0.0.0:8188
To seethe GUI go to: http://[::]:8188
```
打开本地计算机的代理设置，对当前服务器IP设置为不进行代理。
本地浏览器访问 `http://xx.xx.xx.xx:8188`，xx.xx.xx.xx为容器所在服务器IP，即可访问容器内开启的ComfyUI服务。

### 4.2 加载工作流
代码仓中提供stable_audio_open_1.0的加速工作流，路径为`stable_audio_open_1.0/StableAudio_Comfyui_Plugin/workflow.json`，在ComfyUI页面加载该工作流，即可执行推理。
亦可下载ComfyUI官方提供的StableAudio工作流，在ComfyUI页面加载后，右键添加节点，选择`MindIE SD/StableAudio`类别下的加速节点，手动替换官方工作流中模型。
```bash
https://comfyanonymous.github.io/ComfyUI_examples/audio/stable_audio_example.flac
```
### 4.3 执行推理
在UI界面点击运行，完成推理。
