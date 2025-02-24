# SDWebUI-TorchAIE推理指导
torch_aie_extension实现了一个SDWebUI界面的插件，用优化后的diffusers.Unet2DConditionModel替换原有的UNetModel进行推理，支持SD文生图和图生图功能。底层调用了MindIE的build编译优化功能，通过PASS改图、Batch并行等优化手段，提升了推理性能。


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

2. 拉取torch_aie_extension工程，放在stable-diffusion-webui/extensions路径下

3. 获取权重

   ```bash
   # 需要使用 git-lfs (https://git-lfs.com)
   git lfs install
   
   # v1.5，将该权重放在stable-diffusion-webui/extensions/torch_aie_extension/models路径下
   cd stable-diffusion-webui/extensions/torch_aie_extension/models
   git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
   ```
   
4. 将特定权重放在stable-diffusion-webui/models/Stable-diffusion路径下。注意：本插件支持的webui权重如下：

   ```bash
   # 二选一即可，推荐safetensors
   v1-5-pruned-emaonly.safetensors
   v1-5-pruned-emaonly.ckpt
   ```
   
   ```bash
   # 举例：
   cp stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors ../../../models/Stable-diffusion
   ```
   
5. 在stable-diffusion-webui工程路径下执行命令启动webui，自动安装需要的环境

   ```bash
   python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half
   ```

## 运行功能
1. 执行命令启动webui
```bash
python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half --skip-prepare-environment
```
2. 文生图：选择torch_aie按钮，输入文本，设置相关参数，点击generate生成结果

3. 图生图：选择torch_aie按钮，输入图像、文本，设置相关参数，点击generate生成结果

4. 运用并行加速：点击Use_Parallel_Inferencing按钮选择

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。