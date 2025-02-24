# stable-diffusionxl-controlnet模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   ControlNet是一种神经网络架构，可将控制信息添加到预训练的扩散模型中。作用是通过添加额外控制条件，来引导Stable Diffusion生成图像，从而提升 AI 图像生成的可控性和精度。在使用ControlNet模型之后，Stable Diffusion模型的权重被复制出两个相同的部分，分别是“锁定”副本和“可训练”副本。ControlNet主要在“可训练”副本上施加控制条件，然后将施加控制条件之后的结果和原来SD模型的结果相加获得最终的输出结果。神经架构与“零卷积”（零初始化卷积层）连接，参数从零逐渐增长，确保微调的过程不会受到噪声影响。这样可以使用小批量数据集就能对控制条件进行学习训练，同时不会破坏Stable Diffusion模型原本的能力。
   ControlNet的应用包括：控制人物姿势、线稿上色、画质修复等。
- 参考实现：
  ```bash
   # controlnet-canny-sdxl-1.0
   https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0
  ```

- 设备支持：
Atlas 800I A2推理设备：支持的卡数为1

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套   | 版本    | 环境准备指导 |
  | ------ | ------- | ------------ |
  | Python | 3.10.13 | -            |
  | torch  | 2.1.0   | -            |

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```bash
   pip3 install -r requirements.txt
   ```
2. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```
3. 代码修改

   执行命令：

   ```bash
   python3 stable_diffusion_clip_patch.py
   ```

   ```bash
   python3 stable_diffusion_attention_patch.py
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   ControlNet是一个控制预训练图像扩散模型的神经网络，允许输入调节图像，然后使用该调节图像来操控图像生成。调节图像可从官网下载。
   ```bash
   wget https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。
   使用Pytorch导出pt模型，然后使用MindIE推理引擎转换为适配昇腾的模型。

   0. 获取权重（可选）

      可提前下载权重，放到代码同级目录下，以避免执行后面步骤时可能会出现下载失败。

      ```bash
      # 需要使用 git-lfs (https://git-lfs.com)
      git lfs install

      # xl
      git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
      # controlnet-canny-sdxl-1.0
      git clone https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0
      # sdxl-vae-fp16-fix
      git clone https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
      ```
   1. 导出pt模型并进行编译。(可选)
      
      设置模型名称或路径
      ```bash
      # xl (执行时下载权重)
      model_base="stabilityai/stable-diffusion-xl-base-1.0"
      # controlnet-canny-sdxl-1.0 (执行时下载权重)
      model_controlnet="diffusers/controlnet-canny-sdxl-1.0"
      # sdxl-vae-fp16-fix (执行时下载权重)
      model_vae="madebyollin/sdxl-vae-fp16-fix"

      # xl (使用上一步下载的权重)
      model_base="./stable-diffusion-xl-base-1.0"
      # controlnet-canny-sdxl-1.0 (使用上一步下载的权重)
      model_controlnet="./controlnet-canny-sdxl-1.0"
      # sdxl-vae-fp16-fix (使用上一步下载的权重)
      model_vae="./sdxl-vae-fp16-fix"
      ```

      执行命令：
      ```bash
      # 静态模型
      python3 export_ts_controlnet.py --model ${model_base} --controlnet_model ${model_controlnet} --vae_model ${model_vae} --output_dir ./models --batch_size 1 --flag 0 --soc A2 --device 0

      # 动态分档模型，仅支持1024*1024、512*512两种
      python3 export_ts_controlnet.py --model ${model_base} --controlnet_model ${model_controlnet} --vae_model ${model_vae} --output_dir ./models --batch_size 1 --flag 1 --soc A2 --device 0

      ```

      参数说明：

      - --model：模型权重路径
      - --controlnet_model: controlnet模型权重路径
      - --vae_model: vae模型权重路径
      - --output_dir: ONNX模型输出目录
      - --batch_size: 设置batch_size, 默认值为1,当前仅支持batch_size=1的场景
      - --falg: 设置模型编译方式。默认值为1。值为0表示静态模型，值为1表示动态分档模型。
      - --soc: 默认值为A2，当前仅支持Atlas 800I A2场景。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。

      静态编译场景：

      - ./models/clip/clip_bs{batch_size}.pt, ./models/clip/clip_bs{batch_size}_compile.ts 和 ./models/clip/clip2_bs{batch_size}.pt, ./models/clip/clip2_bs{batch_size}_compile.ts
      - ./models/unet/unet_bs{batch_size x 2}.pt, ./models/unet/unet_bs{batch_size x 2}_compile_static.ts
      - ./models/vae/vae_bs{batch_size}.pt, ./models/vae/vae_bs{batch_size}_compile_static.ts
      - ./models/control/control_bs{batch_size}.pt, ./models/control/control_bs{batch_size}_compile_static.ts

      动态分档场景：

      - ./models/clip/clip_bs{batch_size}.pt, ./models/clip/clip_bs{batch_size}_compile.ts 和 ./models/clip/clip2_bs{batch_size}.pt, ./models/clip/clip2_bs{batch_size}_compile.ts
      - ./models/unet/unet_bs{batch_size x 2}.pt, ./models/unet/unet_bs{batch_size x 2}_compile.ts
      - ./models/vae/vae_bs{batch_size}.pt, ./models/vae/vae_bs{batch_size}_compile.ts
      - ./models/control/control_bs{batch_size}.pt, ./models/control/control_bs{batch_size}_compile.ts
  
2. 开始推理验证。

   1. 执行推理脚本。

      ```bash
      python3 stable_diffusionxl_pipeline_controlnet.py \
              --model ${model_base} \
              --controlnet_model ${model_controlnet} \
              --vae_model ${model_vae} \
              --device 0 \
              --save_dir ./results \
              --output_dir ./models \
              --soc A2 \
              --flag 1 \
              --w_h 1024 
      ```

      参数说明：

      - --model：模型名称或本地模型目录的路径。
      - --controlnet_model: controlnet模型权重路径
      - --vae_model: vae模型权重路径
      - --device：推理设备ID。
      - --save_dir：生成图片的存放目录。
      - --output_dir：存放导出模型的目录。
      - --soc: 默认值为A2，当前仅支持Atlas 800I A2场景。
      - --falg: 设置模型编译方式。默认值为1。值为0表示静态模型，值为1表示动态分档模型。
      - --w_h: image的宽高，设置为1024表示宽高均为1024，设置为512表示宽高均为512。仅支持这两种分辨率。
    
      执行完成后在 `./results`目录下生成推理图片。

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。