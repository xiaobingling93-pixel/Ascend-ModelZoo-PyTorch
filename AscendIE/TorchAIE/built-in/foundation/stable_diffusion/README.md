# stable-diffusion模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)
  
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   stable-diffusion是一种文本到图像的扩散模型，能够在给定任何文本输入的情况下生成照片逼真的图像。有关稳定扩散函数的更多信息，请查看[Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion)。

- 参考实现：
  ```bash
   # StableDiffusion v1.5
   https://huggingface.co/runwayml/stable-diffusion-v1-5
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    |  1 x 77 | FLOAT32|  ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 512 x 512 x 3 | FLOAT32  | NHWD           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
- 
  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
   | torch| 2.1.0  | -                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt
   ```

2. 代码修改

   执行命令：
   
   ```bash
   python3 stable_diffusion_clip_patch.py
   ```
   
      ```bash
   python3 stable_diffusion_attention_patch.py
      ```
   
## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息生成图片，无需数据集。
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。【可选】
   使用Pytorch导出pt模型，然后使用MindIE推理引擎转换为适配昇腾的模型。

   0. 获取权重（可选）

       可提前下载权重，以避免执行后面步骤时可能会出现下载失败。

       ```bash
       # 需要使用 git-lfs (https://git-lfs.com)
       git lfs install
       
       # v1.5
       git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
       
       # v2.1
       git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
       ```

   1. 导出pt模型并进行编译。(可选)

      设置模型名称或路径
      ```bash
      # v1.5 (执行时下载权重)
      model_base="runwayml/stable-diffusion-v1-5"
      
      # v1.5 (使用上一步下载的权重)
      model_base="./stable-diffusion-v1-5"
      
      # v2.1 (执行时下载权重)
      model_base="stabilityai/stable-diffusion-2-1-base"
      
      # v2.1 (使用上一步下载的权重)
      model_base="./stable-diffusion-2-1-base"
      ```

      使用Lora权重【可选】

      在[civitai](https://civitai.com)下载base model为SD1.5和SD2.1的的lora权重，一般选择safetensor格式的权重。执行转换脚本，将lora权重和model_base权重结合在一起。

      ```bash
      model_lora=lora权重路径
      model_new=适配lora之后的SD权重路径
      python3 convert_lora_safetensors_to_diffusers.py --base_model_path ${model_base} --checkpoint_path ${model_lora} --dump_path ${model_new}
      ```
   
      注意：若条件允许，该模型可以双芯片并行的方式进行推理，从而获得更短的端到端耗时。具体指令的差异之处会在后面的步骤中单独说明，请留意。
   
      执行命令：
   
      ```bash
      # 若不使用lora权重
      python3 export_ts.py --model ${model_base} --output_dir ./models
      
      # 若使用lora权重
      python3 export_ts.py --model ${model_new} --output_dir ./models_lora
      ```
   
      参数说明：
      - --model：模型名称或本地模型目录的路径
      - --output_dir: pt模型输出目录
      
      执行成功后会生成pt模型：  
         - ./models/clip/clip.pt  或者 ./models_lora/clip/clip.pt 
         - ./models/unet/unet_bs1.pt 或者 ./models_lora/unet/unet_bs1.pt
         - ./models/unet/unet_bs2.pt 或者 ./models_lora/unet/unet_bs2.pt
         - ./models/vae/vae.pt 或者 ./models_lora/vae/vae_bs2.pt
      
   2. **注意：**更换lora权重时，请手动删除models_lora路径的生成的pt模型，重新执行转换权重脚本和导出模型命令导出带lora权重的pt模型。


2. 开始推理验证。【Duo】
   1. 执行推理脚本。
      ```bash
      # 1.若不使用并行推理：
      # 1.1不使用lora权重
      python3 stable_diffusion_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --scheduler DDIM \
              --soc Duo \
              --output_dir ./models
       # 1.2使用带lora权重的新权重
       python3 stable_diffusion_pipeline.py \
              --model ${model_new} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --scheduler DDIM \
              --soc Duo \
              --output_dir ./models_lora
              
      # 2.若使用并行推理
      # 2.1不使用lora权重
      python3 stable_diffusion_paralle_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --steps 50 \
              --scheduler DDIM \
              --soc Duo \
              --output_dir ./models
       # 2.2使用带lora权重的新权重
       python3 stable_diffusion_paralle_pipeline.py \
              --model ${model_new} \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --steps 50 \
              --scheduler DDIM \
              --soc Duo \
              --output_dir ./models_lora
      ```
      
      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --prompt_file：输入文本文件，按行分割。
      - --save_dir：生成图片的存放目录。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
      - --scheduler: 采样器。
      - --soc: 硬件配置，根据硬件配置选择Duo或者A2。默认是Duo。
      - --output_dir: 编译好的模型路径。
      
      执行完成后在`./results`目录下生成推理图片。并在终端显示推理时间。


2. **注意**：更换lora权重时，请手动删除models_lora路径的生成的编译好的pt模型，（xxx_compile.pt）重新执行推理脚本。
   
   
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

性能参考下列数据。

### StableDiffusion v1.5

| 硬件形态 | 迭代次数 | 平均耗时 |
| :------: |:----:|:----:|
| Duo并行  |  50  | 2.8s |
| A2     |  50  |  2s  | 
