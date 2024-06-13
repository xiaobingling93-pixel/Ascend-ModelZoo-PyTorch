# stable-diffusionxl模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)
  
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   SDXL 由一组用于潜在扩散的专家管道组成： 在第一步中，使用基础模型生成（噪声）潜伏， 然后使用专门用于最终降噪步骤的细化模型[此处获得](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/)

- 参考实现：
  ```bash
   # StableDiffusionxl
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | prompt    |  1 x 77 | INT64|  ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 3 x 1024 x 1024 | FLOAT32  | NCHW          |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
   | torch| 2.1.0  | -                                                            |

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt
   ```

2. 安装mindie和mindietorch包

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
   
   ```bash
   # 若使用unetCache
   python3 stable_diffusionxl_unet_patch.py
   ```
   
## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息生成图片，无需数据集。


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
       ```

   1. 导出pt模型并进行编译。(可选)

      ```bash
      # xl (执行时下载权重)
      model_base="stabilityai/stable-diffusion-xl-base-1.0"
      
      xl (使用上一步下载的权重)
      model_base="./stable-diffusion-xl-base-1.0"
      ```

      执行命令：
   
      ```bash
      # 使用unetCache, 非并行
      python3 export_ts.py --model ${model_base} --output_dir ./models --use_cache --batch_size 1 --flag 0 --soc A2 --device 0
      
      # 使用unetCache, 并行
      python3 export_ts.py --model ${model_base} --output_dir ./models --use_cache --parallel --batch_size 1 --flag 0 --soc Duo --device 0
      ```
      参数说明：
      - --model：模型权重路径
      - --output_dir: ONNX模型输出目录
      - --use_cache: 【可选】在推理过程中使用cache
      - --parallel: 【可选】导出适用于并行方案的模型，当前仅带unetCache优化时，支持并行
      - --batch_size: 设置batch_size, 默认值为1,当前仅支持batch_size=1的场景
      - --flag：默认为0。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512；2代表动态shape，height的范围为[512, 1024]，width的范围是[512, 1664]。
      - --soc：只支持Duo和A2。默认为A2
      - --device：推理设备ID
   
2. 开始推理验证。

   1. 执行推理脚本。
      ```bash
      # 不使用unetCache策略
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024
      
      # 使用UnetCache策略
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results_unetCache \
              --steps 50 \
              --output_dir ./models \
              --use_cache \
              --flag 0 \
              --height 1024 \
              --width 1024
      
      # 使用UnetCache策略,同时使用双卡并行策略
      python3 stable_diffusionxl_pipeline_cache_parallel.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results_unetCache_parallel \
              --steps 50 \
              --output_dir ./models \
              --use_cache \
              --flag 0 \
              --height 1024 \
              --width 1024
      ```
      
      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --output_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
      - --use_cache: 【可选】在推理过程中使用cache。
      - --cache_steps: 使用cache的迭代次数，迭代次数越多性能越好，但次数过多可能会导致精度下降。
      - --flag：默认为0。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512；2代表动态shape，height的范围为[512, 1024]，width的范围是[512, 1664]。**注意**：请与导出模型时设置的flag保持一致
      - --height：与flag标志位对应的height一致
      - --width：与flag标志位对应的width一致
      
      不带unetCache策略，执行完成后在`./results`目录下生成推理图片。并在终端显示推理时间，参考如下：
   
      ```
      [info] infer number: 16; use time: 150.567s; average time: 9.410s
      ```
      
      带unetCache策略，执行完成后在`./results_unetCache`目录下生成推理图片。并在终端显示推理时间，参考如下：
      ```
      [info] infer number: 16; use time: 71.855s; average time: 4.491s
      ```
      
      带unetCache策略，同时使用双卡并行策略，执行完成后在`./results_unetCache_parallel`目录下生成推理图片。并在终端显示推理时间，参考如下：
      ```
      [info] infer number: 16; use time: 47.351s; average time: 2.959s
      ```

## 精度验证<a name="section741711594518"></a>

   由于生成的图片存在随机性，所以精度验证将使用CLIP-score来评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。

   注意，由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

   1. 下载Parti数据集

      ```bash
      wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
      ```

   2. 下载Clip模型权重

      ```bash
      GIT_LFS_SKIP_SMUDGE=1 
      git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
      cd ./CLIP-ViT-H-14-laion2B-s32B-b79K
      ```
      也可手动下载[权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)
      将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下

   3. 使用推理脚本读取Parti数据集，生成图片

      ```bash
      # 不使用unetCache策略
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024
      
      # 使用UnetCache策略
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts_unetCache \
              --steps 50 \
              --output_dir ./models \
              --use_cache \
              --flag 0 \
              --height 1024 \
              --width 1024
      
      # 使用UnetCache策略,同时使用双卡并行策略
      python3 stable_diffusionxl_pipeline_cache_parallel.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0,1 \
              --save_dir ./results_PartiPrompts_unetCache_parallel \
              --steps 50 \
              --output_dir ./models \
              --use_cache \
              --flag 0 \
              --height 1024 \
              --width 1024
      
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --output_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --prompt_file_type: prompt文件类型，用于指定读取方式。
      - --num_images_per_prompt: 每个prompt生成的图片数量。
      - --max_num_prompts：限制prompt数量为前X个，0表示不限制。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。

      不带unetCache，执行完成后会在`./results_PartiPrompts`目录下生成推理图片，并且会在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。
      带unetCache，执行完成后会在`./results_PartiPrompts_unetCache`目录下生成推理图片，并且会在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。
      带unetCache，同时使用双卡并行策略，执行完成后会在`./results_PartiPrompts_unetCache_parallel`目录下生成推理图片，并且会在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。

   4. 计算CLIP-score

      ```bash
      python clip_score.py \
             --device=cpu \
             --image_info="image_info.json" \
             --model_name="ViT-H-14" \
             --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
      ```

      参数说明：
      - --device: 推理设备。
      - --image_info: 上一步生成的`image_info.json`文件。
      - --model_name: Clip模型名称。
      - --model_weights_path: Clip模型权重文件路径。

      执行完成后会在屏幕打印出精度计算结果。


## 量化功能【可选】<a name="section741711594518"></a>

若使用W8A8量化功能，分辨率只支持1024x1024和512x512：

   1. 导出模型，height只支持1024和512，width只支持1024和512

      ```bash
      # 使用unetCache, 非并行
      python3 export_ts.py --model ${model_base} --output_dir ./models --use_cache --batch_size 1 --flag 0 --soc A2 --device 0 --height 1024 --width 1024
      
      # 不使用unetCache, 非并行
      python3 export_ts.py --model ${model_base} --output_dir ./models --batch_size 1 --flag 0 --soc A2 --device 0 --height 1024 --width 1024
      ```

   2. 量化编译。./quant/build.sh中的TorchPath需要指定为python安装torch的路径。

      执行命令：

      ```bash
      cd quant
      bash build.sh
      ```

   3. 导出unet pt模型的输入。

      执行命令：

      ```bash
      # 若使用UnetCache策略
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results_temp \
              --steps 50 \
              --output_dir ./models \
              --use_cache \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --save_unet_input
      # 若不使用UnetCache策略
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results_temp \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --save_unet_input
      ```

   4. 导出pt模型并进行编译。

      执行命令：

      ```bash
      # 若使用unetCache, 且非并行
      python3 export_ts_quant.py --model ${model_base} --output_dir ./models_quant --use_cache --batch_size 1 --soc A2 --device 0 --height 1024 --width 1024
      
      # 若不使用unetCache, 且非并行
      python3 export_ts_quant.py --model ${model_base} --output_dir ./models_quant --batch_size 1 --soc A2 --device 0 --height 1024 --width 1024
      ```

   5. 开始推理验证。

      执行命令：

      ```bash
      # 使用UnetCache策略，且非并行
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results_quant \
              --steps 50 \
              --output_dir ./models_quant \
              --flag 3 \
              --use_cache \
              --height 1024 \
              --width 1024 \
              --quant
      # 不使用UnetCache策略，且非并行
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results_quant \
              --steps 50 \
              --output_dir ./models_quant \
              --flag 3 \
              --height 1024 \
              --width 1024 \
              --quant
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

### StableDiffusionxl

| 硬件形态 | 迭代次数 | 平均耗时    | cpu规格 |
| :------: | :--: | :--------: | :--------: |
| A2  |    50  |  6.542s   | 64核(arm) |

性能测试需要独占npu和cpu

迭代50次的参考精度结果如下：

   ```
   average score: 0.378
   category average scores:
   [Abstract], average score: 0.265
   [Vehicles], average score: 0.380
   [Illustrations], average score: 0.372
   [Arts], average score: 0.414
   [World Knowledge], average score: 0.391
   [People], average score: 0.379
   [Animals], average score: 0.390
   [Artifacts], average score: 0.373
   [Food & Beverage], average score: 0.372
   [Produce & Plants], average score: 0.370
   [Outdoor Scenes], average score: 0.373
   [Indoor Scenes], average score: 0.389
   ```