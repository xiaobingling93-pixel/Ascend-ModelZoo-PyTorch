# SDWebUI-TorchAIE推理指导
torch_aie_extension实现了一个SDWebUI界面的插件，用优化后的diffusers.Unet2DConditionModel替换原有的UNetModel进行推理，支持SD文生图和图生图功能。

# 概述

   SDWebUI是一个基于Gradio库的WebUi界面，支持设置输入和参数用于SD模型的文生图、图生图等功能。有关SDWebUI的更多信息，请查看[Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)。

# 推理环境准备

该插件依赖torch2.1.0, python3.10环境

# 快速上手
## sd_webui部署
1. 拉取代码[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   ```

2. 拉取onnx_extension工程，放在stable-diffusion-webui/extensions路径下

3. 获取权重

   ```bash
   # 需要使用 git-lfs (https://git-lfs.com)
   git lfs install
   
   # v2.1,将该权重放在stable-diffusion-webui/extensions/onnx_extension/models路径下
   cd stable-diffusion-webui/extensions/onnx_extension/models
   git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
   
   # 将stable-diffusion-2-1-base下的v2-1_512-ema-pruned.safetensors复制到stable-diffusion-webui/models/Stable-diffusion路径下
   cp stable-diffusion-2-1-base/v2-1_512-ema-pruned.safetensors ../../../models/Stable-diffusion
   cd ../../..
   ```

4. 在webui工程路径下执行命令启动webui，自动安装需要的环境

   ```bash
   python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half
   ```

## 插件部署
1. 按照requirements.txt要求的版本安装相关依赖，避免导出模型失败！
```bash
   pip install -r requirements.txt
```
2. 安装昇腾推理工具

   请访问[mist代码仓](https://gitee.com/ascend/msit/tree/master/msit/)，根据readme文档进行工具安装。可只安装需要的组件：debug surgeon，其他组件为可选安装。
   
   请访问[ais_bench](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，根据readme文件进行工具安装，建议使用whl包进行安装。

2. 代码修改，修改clip和cross_attention，用于导出正确的模型
```bash
   python sd_webui_patch.py
```
3. 安装aie包和torch_aie包，配置AIE目录下的环境变量
```bash
   chmod +x ./Ascend-cann-aie_xxx.run
   ./Ascend-cann-aie_xxx.run --install
   source set_env.sh
```

## 运行功能
1. 执行命令启动webui
```bash
python launch.py --skip-torch-cuda-test --port 22 --enable-insecure-extension-access --listen --log-startup --disable-safe-unpickle --no-half --skip-prepare-environment
```
2. 请优先选择device，310P3选择Duo，910B4选择A2
3. 文生图：选择ONNX按钮，输入文本，设置相关参数，点击generate生成结果
4. 图生图：选择ONNX按钮，输入图像、文本，设置相关参数，点击generate生成结果
5. 运用并行加速：点击Use_Parallel_Inferencing按钮选择

# 备注

1. 使用昇腾插件后，原始的webui界面中的某些配置受到限制，如下：

   可配置参数：

   ```
   Sampling method
   Sampling steps
   CFG Scale
   Seed
   ```

   受限制参数：

   ```
   Width和Height要固定为512
   Batch count要固定为1
   Batch size要固定为1
   ```

2. 点击ONNX按钮，在第一次启动服务后，会做模型的处理，该处理会耗时10分钟左右，当后台输出"You can generate image now!"字样时，可进行图生成等操作。
