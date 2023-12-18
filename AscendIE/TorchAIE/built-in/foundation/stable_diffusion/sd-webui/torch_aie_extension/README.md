# SDWebUI-TorchAIE推理指导
torch_aie_extension实现了一个SDWebUI界面的插件，用优化后的diffusers.Unet2DConditionModel替换原有的UNetModel进行推理，支持SD文生图和图生图功能。底层调用了AscendIE的build编译优化功能，通过PASS改图、Batch并行等优化手段，提升了推理性能。


# 概述

   SDWebUI是一个基于Gradio库的WebUi界面，支持设置输入和参数用于SD模型的文生图、图生图等功能。有关SDWebUI的更多信息，请查看[Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)。

# 推理环境准备

该插件依赖torch2.1, python3.10环境

# 快速上手
## sd_webui部署
1. 拉取代码[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)，执行命令启动webui，自动安装需要的环境
```bash
python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half
```
## 插件部署
1. 提前拉取torch_aie_extension代码，放到sd_webui中extension目录下，请按照requirements.txt要求的版本安装相关依赖，避免导出模型失败！
```bash
   pip install -r requirements.txt
   ```
2. 代码修改，修改clip和cross_attention，用于trace正确的模型
```bash
   python sd_webui_patch.py
   ```
3. 安装aie包和torch_aie包，配置AIE目录下的环境变量
```bash
   ./Ascend-cann-aie_xxx.run --install-path=/home/xxx
   source set_env.sh
   pip install torch_aie-xxx.whl --force-reinstall
   ```

## 运行功能
1. 执行命令启动webui
```bash
python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half --skip-prepare-environment
```
2. 文生图：选择torch_aie按钮，输入文本，设置相关参数，点击generate生成结果

3. 图生图：选择torch_aie按钮，输入图像、文本，设置相关参数，点击generate生成结果

4. 运用并行加速：点击Use_Parallel_Inferencing按钮选择，仅支持310P，910B未支持
