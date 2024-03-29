# SDWebUI-TorchAIE推理指导
mindie_extension实现了一个SDWebUI界面的插件，用优化后的diffusers.Unet2DConditionModel替换原有的UNetModel进行推理，支持SD文生图和图生图功能。底层调用了MindIE的build编译优化功能，通过PASS改图、Batch并行等优化手段，提升了推理性能。


# 概述

   SDWebUI是一个基于Gradio库的WebUi界面，支持设置输入和参数用于SD模型的文生图、图生图等功能。有关SDWebUI的更多信息，请查看[Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)。

# 推理环境准备

该插件依赖torch2.1.0, python3.10环境

# 快速上手

## 环境准备

1. 按照requirements.txt要求的版本安装相关依赖，避免导出模型失败！

   ```
   pip install -r requirements.txt
   ```

2. 安装mindie包和mindietorch包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/aie/set_env.sh
   # 安装mindietorch
   tar -zxvf Ascend-mindie-torch_xxx.tar.gz
   pip install mindietorch-1.0.rc1+torch2.1.0xxx.whl
   ```

3. 代码修改，修改clip和cross_attention，用于trace正确的模型

   ```bash
   python sd_webui_patch.py
   ```

## sd_webui部署

1. 拉取webui工程代码[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   ```

2. 拉取mindie_extension工程，放在stable-diffusion-webui/extensions路径下

3. 将mindie_extension工程的diff_1.patch和diff_2.patch放到stable-diffusion-webui路径下

   ```bash
   mv diff_1.patch diff_2.patch ../..
   patch -p0 < diff_1.patch
   patch -p0 < diff_2.patch
   ```

4. 获取权重

   ```bash
   # 需要使用 git-lfs (https://git-lfs.com)
   git lfs install
   
   # v1.5，将该权重放在stable-diffusion-webui/extensions/mindie_extension/models路径下
   cd stable-diffusion-webui/extensions/torch_aie_extension/models
   git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
   
   # v2.1，将该权重放在stable-diffusion-webui/extensions/mindie_extension/models路径下
   git clone https://huggingface.co/runwayml/stable-diffusion-2-1-base
   
   # sdxl，将该权重放在stable-diffusion-webui/extensions/mindie_extension/models路径下
   git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
   ```

5. 将特定权重放在stable-diffusion-webui/models/Stable-diffusion路径下。

   注意：本插件支持的webui权重如下：

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

6. 在stable-diffusion-webui工程路径下执行命令启动webui，自动安装需要的环境

   ```bash
   python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half
   ```

## 运行功能
1. 执行命令启动webui
```bash
python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half --skip-prepare-environment
```
2. 使用该插件后，原始的webui界面中的某些配置受到限制，如下：

   可配置参数：

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

3. 界面启动后，请先选择硬件配置，310P3请选择Duo，910B4请选择A2。然后选择MindIE_torch按钮，第一次启动服务时，点击MindIE_torch按钮后，会对于原始模型做一些处理，请耐心等待，直到服务端显示"You can generate image now!"字样后，再根据上述参数配置，点击generate生成结果。

