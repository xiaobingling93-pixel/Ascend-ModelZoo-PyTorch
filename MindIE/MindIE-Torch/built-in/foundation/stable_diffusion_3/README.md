# stable-diffusion3模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)
  
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   SD3 由一组用于潜在扩散的专家管道组成： 在第一步中，使用基础模型生成（噪声）潜伏， 然后使用专门用于最终降噪步骤的细化模型[此处获得](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/)

- 参考实现：
  ```bash
   # StableDiffusion3
   https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
  ```

- 设备支持：
Atlas 800I A2推理设备：支持的卡数为1或2
Atlas 300I Duo推理卡：支持的卡数为1，可双芯并行

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | prompt    |  1 x 77 | INT64|  ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ----------- |
  | output1  | 1 x 3 x 1024 x 1024 | FLOAT32  | NCHW |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10.2 | - |
  | torch | 2.1.0 | - |

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 获取源码<a name="section4622531142816"></a>
1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt

   # 若要使用hpsv2验证精度，则还需要按照以下步骤安装hpsv2
   git clone https://github.com/tgxs002/HPSv2.git
   cd HPSv2
   pip3 install -e .
   ```
   - 注意：当前sd3推理暂不支持mindie与torch_npu混用，请确保实际推理环境中没有安装torch_npu

2. 安装mindie包

   ```bash
   # 安装mindie
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```
   
3. 代码修改（可选）
（1）若需要开启DiTCache、序列压缩等优化，需要执行以下代码修改操作：
- 若环境没有patch工具，请自行安装：
   ```bash
    apt update
    apt install patch
   ```
- 执行命令：
   ```bash
   python3 attention_patch.py
   python3 attention_processor_patch.py
   python3 transformer_sd3_patch.py
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。
   使用Pytorch导出pt模型，然后使用MindIE推理引擎转换为适配昇腾的模型。

   0. 获取权重（可选）

       可提前下载权重，放到代码同级目录下，以避免执行后面步骤时可能会出现下载失败。

       ```bash
       # 需要使用 git-lfs (https://git-lfs.com)
       git lfs install
       
       # 下载sd3权重
       git clone https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
       ```

   1. 导出pt模型并进行编译。
      (1) 设置模型权重的路径
      ```bash
      # sd3 (执行时下载权重)
      model_base="stabilityai/stable-diffusion-3-medium-diffusers"
      
      # sd3 (使用上一步下载的权重)
      model_base="./stable-diffusion-3-medium-diffusers"
      ```
      (2) 创建文件夹./models存放导出的模型
      ```bash
      mkdir ./models
      ```
      (3) 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         ```

      (4) 执行export命令
   
      ```bash
      # Atlas 800I A2，非并行，未加DiTCache优化
      python3 export_model.py --model ${model_base} --output_dir ./models --batch_size 1 --soc Ascend${chip_name} --device_type A2 --device 0
      # Atlas 800I A2，非并行。开启DiTCache优化
      python3 export_model.py --model ${model_base} --output_dir ./models --batch_size 1 --soc Ascend${chip_name} --device_type A2 --device 0 --use_cache
      
      # Atlas 300I Duo，并行
      python3 export_model.py --model ${model_base} --output_dir ./models --parallel --batch_size 1 --soc Ascend${chip_name} --device_type Duo --device 0
      ```
      参数说明：
      - --model：模型权重路径
      - --output_dir: 存放导出模型的路径
      - --parallel: 【可选】导出适用于并行方案的模型
      - --batch_size: 设置batch_size, 默认值为1, 当前仅支持batch_size=1的场景
      - --soc：处理器型号。
      - --device_type: 设备形态，当前支持A2、Duo两种形态。
      - --device：推理设备ID
      - --use_cache：开启DiTCache优化，不配置则不开启
      注意：trace+compile耗时较长且占用较多的CPU资源，请勿在执行export命令时运行其他占用CPU内存的任务，避免程序意外退出。
   
2. 开始推理验证。

   1. 开启cpu高性能模式
      ```bash
      echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
      sysctl -w vm.swappiness=0
      sysctl -w kernel.numa_balancing=0
      ```

   2. 安装绑核工具
      ```bash
      apt-get update
      apt-get install numactl
      ```
      查询卡的NUMA node
      ```shell
      lspci -vs bus-id
      ```
      bus-id可通过npu-smi info获得，查询到NUMA node，在推理命令前加上对应的数字

      可通过lscpu获得NUMA node对应的CPU核数
      ```shell
      NUMA node0: 0-23
      NUMA node1: 24-47
      NUMA node2: 48-71
      NUMA node3: 72-95
      ```
      当前查到NUMA node是0，对应0-23，推荐绑定其中单核以获得更好的性能。
   
   3. 执行推理脚本。
      ```bash
      # 不使用DiTCache，单卡推理，适用Atlas 800I A2场景
      numactl -C 0-23 python3 stable_diffusion3_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --prompt_file_type plain \
              --device 0 \
              --save_dir ./results \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1
      
      # 不使用DiTCache，使用双卡并行推理，适用Atlas 300I DUO场景
      numactl -C 0-23 python3 stable_diffusion3_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --prompt_file_type plain \
              --device 0,1 \
              --save_dir ./results_parallel \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1
      
      # 使用DiTCache，单卡推理，适用Atlas 800I A2场景
      numactl -C 0-23 python3 stable_diffusion3_pipeline_cache.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --prompt_file_type plain \
              --device 0 \
              --save_dir ./results \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1 \
              --use_cache
      ```
      
      参数说明：
      - --model：模型权重路径。
      - --output_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
      - --height：生成图像高度，当前只支持1024
      - --width：生成图像宽度，当前只支持1024
      - --use_cache：开启DiTCache优化，不配置则不开启
      
      非并行策略，执行完成后在`./results`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。
      并行策略，同时使用双卡并行策略，执行完成后在`./results_parallel`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。并在终端显示推理时间。
      注意：当前MindIE-Torch和torch_npu的synchronizing stream不兼容，为避免出错，建议在运行推理前先卸载torch_npu。

## 精度验证<a name="section741711594518"></a>

   由于生成的图片存在随机性，提供两种精度验证方法：
   1. CLIP-score（文图匹配度量）：评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。使用Parti数据集进行验证。
   2. HPSv2（图片美学度量）：评估生成图片的人类偏好评分，分数的取值范围为[0, 1]，越高越好。使用HPSv2数据集进行验证

   注意，由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

   1. 下载Parti数据集和hpsv2数据集

      ```bash
      # 下载Parti数据集
      wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
      ```
      hpsv2数据集下载链接：https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_benchmark_prompts.json

   2. 下载模型权重

      ```bash
      # Clip Score和HPSv2均需要使用的权重
      GIT_LFS_SKIP_SMUDGE=1
      git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
      cd ./CLIP-ViT-H-14-laion2B-s32B-b79K
      
      # HPSv2权重
      wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt --no-check-certificate
      ```
      也可手动下载[权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)
      将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下，手动下载[HPSv2权重](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)放到当前路径

   3. 使用推理脚本读取Parti数据集，生成图片

      ```bash
      # 不使用并行
      python3 stable_diffusion3_pipeline.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1
      
      # 使用DitCache
      python3 stable_diffusion3_pipeline_cache.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1 \
              --use_cache

      # 使用双卡并行策略
      python3 stable_diffusion3_pipeline.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0,1 \
              --save_dir ./results_PartiPrompts_parallel \
              --steps 28 \
              --output_dir ./models \
              --use_cache \
              --height 1024 \
              --width 1024 \
              --batch_size 1
      ```

      参数说明：
      - --model：模型权重路径。
      - --output_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。注意使用hpsv2时，设置num_images_per_prompt=1即可。
      - --num_images_per_prompt: 每个prompt生成的图片数量。注意使用hpsv2时，设置num_images_per_prompt=1即可。
      - --max_num_prompts：限制prompt数量为前X个，0表示不限制。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。

      不使用并行策略，执行完成后在`./results_PartiPrompts`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。
      使用双卡并行策略，执行完成后在`./results_PartiPrompts_parallel`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。并在终端显示推理时间。

4. 使用推理脚本读取hpsv2数据集，生成图片

      ```bash
      # 不使用并行
      python3 stable_diffusion3_pipeline.py \
              --model ${model_base} \
              --prompt_file_type hpsv2 \
              --num_images_per_prompt 1 \
              --info_file_save_path ./image_info_hpsv2.json \
              --device 0 \
              --save_dir ./results_hpsv2 \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1

      # 使用DitCache
      python3 stable_diffusion3_pipeline_cache.py \
              --model ${model_base} \
              --prompt_file_type hpsv2 \
              --num_images_per_prompt 1 \
              --info_file_save_path ./image_info_hpsv2.json \
              --device 0 \
              --save_dir ./results_hpsv2 \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1 \
              --use_cache
      
      # 使用双卡并行策略
      python3 stable_diffusion3_pipeline.py \
              --model ${model_base} \
              --prompt_file_type hpsv2 \
              --num_images_per_prompt 1 \
              --info_file_save_path ./image_info_hpsv2.json \
              --device 0,1 \
              --save_dir ./results_hpsv2_parallel \
              --steps 28 \
              --output_dir ./models \
              --height 1024 \
              --width 1024 \
              --batch_size 1
      ```
   参数说明：
      - --info_file_save_path：生成图片信息的json文件路径。

      不使用并行策略，执行完成后在`./results_hpsv2`目录下生成推理图片，在当前目录生成一个`image_info_hpsv2.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。
      使用双卡并行策略，执行完成后在`./results_hpsv2_parallel`目录下生成推理图片，在当前目录生成一个`image_info_hpsv2.json`文件，记录着图片和prompt的对应关系。并在终端显示推理时间。

5. 计算精度指标
      1. CLIP-score
         ```bash
         python3 clip_score.py \
               --device=cpu \
               --image_info="image_info.json" \
               --model_name="ViT-H-14" \
               --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
         ```

         参数说明：
         - --device: 推理设备，默认为"cpu"，如果是cuda设备可设置为"cuda"。
         - --image_info: 上一步生成的`image_info.json`文件。
         - --model_name: Clip模型名称。
         - --model_weights_path: Clip模型权重文件路径。

         clip_score.py脚本可参考[SDXL](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/clip_score.py)，执行完成后会在屏幕打印出精度计算结果。
      
      2. HPSv2
         ```bash
         python3 hpsv2_score.py \
               --image_info="image_info_hpsv2.json" \
               --HPSv2_checkpoint="./HPS_v2_compressed.pt" \
               --clip_checkpoint="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
         ```

         参数说明：
         - --image_info: 上一步生成的`image_info_hpsv2.json`文件。
         - --HPSv2_checkpoint: HPSv2模型权重文件路径。
         - --clip_checkpointh: Clip模型权重文件路径。

         hpsv2_score.py脚本可参考[SDXL](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_score.py)，执行完成后会在屏幕打印出精度计算结果。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

### StableDiffusion3
| 硬件形态  | cpu规格 | batch size | 迭代次数 | 优化手段 | 平均耗时  |        精度        |
| :------: | :------: | :------: |:----:| :------: |:-----:|:----------------:|
| Atlas 800I A2 (32G) | 64核(arm) |  1  |  28  | w/o UnetCache | 6.15s | clip score 0.380 |

性能测试需要独占npu和cpu