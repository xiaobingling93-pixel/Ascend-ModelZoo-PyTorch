# ControlNet模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)
   
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>
   ControlNet是一种神经网络架构，可将控制信息添加到预训练的扩散模型中。作用是通过添加额外控制条件，来引导Stable Diffusion生成图像，从而提升 AI 图像生成的可控性和精度。在使用ControlNet模型之后，Stable Diffusion模型的权重被复制出两个相同的部分，分别是“锁定”副本和“可训练”副本。ControlNet主要在“可训练”副本上施加控制条件，然后将施加控制条件之后的结果和原来SD模型的结果相加获得最终的输出结果。神经架构与“零卷积”（零初始化卷积层）连接，参数从零逐渐增长，确保微调的过程不会受到噪声影响。这样可以使用小批量数据集就能对控制条件进行学习训练，同时不会破坏Stable Diffusion模型原本的能力。如今ControlNet的应用包括：控制人物姿势、线稿上色、画质修复等。

- 参考实现：
  ```
  url=https://github.com/lllyasviel/ControlNet 
  branch=main
  commit_id=ed85cd1e25a5ed592f7d8178495b4483de0331bf
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | text    |  1 x 4 x 64 x 72 | FLOAT32 |  NCHW|
  | hint    |  1 x 3 x 512 x 576 | FLOAT32 | NCHW|
  | t       |  1                | INT64 | ND|
  | cond_text| 1 x 77 x 768     | FLOAT32| ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | text_outs | 1 x 4 x 64 x 72 | FLOAT32  | NCHW           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.rc3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 7.0.0.B021 | -                                                            |
  | Python                                                       | 3.10.6   | -                                                            |                                                           |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码
   ```
   git clone https://github.com/lllyasviel/ControlNet
   mv export2pt.py controlnet_infer.py torch_aie_cldm.py diff.patch ControlNet/
   ```

2. 安装依赖。
   ```
   pip3 install -r requirements.txt
   ```

3. 代码修改

   执行命令：
   
   ```
   cd ControlNet
   patch -p1 < diff.patch
   ```

4. 安装aie包和torch_aie包，配置AIE目录下的环境变量
```bash
   ./Ascend-cann-aie_xxx.run --install-path=/home/xxx
   source set_env.sh
   pip install torch_aie-xxx.whl --force-reinstall
   ```
   
   
## 模型推理<a name="section741711594517"></a>

   1. 获取权重

     
      ```
      训练权重链接为："https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_canny.pth"。
      下载后放入`ControlNet/base_model`工作目录下.
      ```

   2. 导出PT模型

      
      ```
      可提前下载openclip权重放入'ControlNet/openai/clip-vit-large-patch14'，以避免执行后面步骤时可能会出现下载失败。
      # 需要使用 git-lfs (https://git-lfs.com)
      git lfs install
      cd ControlNet
      git clone https://huggingface.co/openai/clip-vit-large-patch14
      ```
      请在python3.8和torch1.12.1等requirements.txt安装版本下trace导出PT模型。
      

      执行命令：

      ```
      python export2pt.py --model ./base_models/control_sd15_canny.pth --control_path ./models  --sd_path ./models
      ```

      参数说明：
      - --base_model：模型权重的路径
      - --control_path: control部分PT模型输出目录
      - --sd_path：sd部分PT模型输出目录

   3. 模型推理
      请在python3.10和torch2.1环境下进行模型编译优化和推理，执行推理脚本。
      ```
    
      python3 controlnet_infer.py \
              --base_model ./base_models/control_sd15_canny.pth \
              --image test_imgs/dog.png \
              --prompt "cute dog" \
              --device 0 \
              --control_model_dir ./models/control/control.pt \
              --sd_model_dir ./models/sd/sd.pt \
              --save_dir ./results \
              --ddim_steps 20 
   
      ```

      参数说明：
      - --base_model：模型权重的路径。
      - --prompt：文本信息。
      - --save_dir：生成图片的存放目录。
      - --ddim_steps：生成图片次数。
      - --image: 输入图片。
      - --control_model_dir: control的pt模型位置。
      - --sd_model_dir: sd的pt模型位置。
      - --device：推理设备ID。
      
      执行完成后在`./results`目录下生成推理图片。推理一张图片会输出一张图片边缘的图片，和一张跟据输入图片和文本重新生成的图片。
