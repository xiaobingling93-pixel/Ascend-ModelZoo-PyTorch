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
  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 24.1.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN（+MindIE-RT）                                              | 8.0.RC1(1.0.RC1) | -                                                            |
  | Python                                                       | 3.10   | -                                                            |                                                           |
如在优化模型时使用了--FA、--TOME_num、--faster_gelu参数，需要安装与CANN包配套版本的MindIE-RT

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt
   ```

2. 代码修改

   执行命令：
   
   ```bash
   TRANSFORMERS_PATH=`python3 -c "import transformers; print(transformers.__path__[0])"`
   patch  -p0 ${TRANSFORMERS_PATH}/models/clip/modeling_clip.py clip.patch 
   ```

3. 安装昇腾统一推理工具（AIT）

   请访问[AIT代码仓](https://gitee.com/ascend/ait/tree/master/ait#ait)，根据readme文档进行工具安装。可只安装需要的组件：debug surgeon，其他组件为可选安装。
   
   请访问[ais_bench](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，根据readme文件进行工具安装。
   

## 模型推理<a name="section741711594517"></a>

1. 模型转换。
   使用PyTorch将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   0. 获取权重（可选）

      可提前下载权重，放到代码同级目录下，以避免执行后面步骤时可能会出现下载失败。

      ```bash
      # 需要使用 git-lfs (https://git-lfs.com)
      git lfs install

      # 下载权重
      git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
      ```

   1. 导出ONNX模型

      设置模型名称或路径
      ```bash
      # base (执行时下载权重)
      model_base="stabilityai/stable-diffusion-xl-base-1.0"

      # base (下载的权重路径)
      model_base="./stable-diffusion-xl-base-1.0"
      ```

      执行命令：

      ```bash
      python3 stable_diffusionxl_2_onnx.py --model ${model_base} --output_dir ./models

      ```

      参数说明：
      - --model：模型权重路径
      - --output_dir: ONNX模型输出目录
 
      
      执行成功后生成onnx模型：
         ```
         |—— models
                |—— text_encoder 
                       |—— text_encoder.onnx 
                       |—— text_encoder_2.onnx 
                |—— unet 
                       |—— unet.onnx 
                |—— vae 
                       |—— vae.onnx 
                |—— ddim 
                       |—— ddim.onnx 
         ```      

   2. 优化onnx模型

      1. 量化（可选，可提升性能但可能导致精度下降）

         量化步骤请参考[量化指导](./README_quant.md)

      2. 模型优化

         运行modify_onnx.py脚本。
         ```bash 
         bs=1
         # 非并行方案
         python3 modify_onnx.py \
               --model models/unet/unet.onnx \
               --new_model models/unet/unet_md.onnx \
               --FA_soc A2 \
               --TOME_num 10 \
               --faster_gelu \
               --batch_size ${bs}
         
         # 并行方案
         python3 modify_onnx.py \
               --model models/unet/unet.onnx \
               --new_model models/unet/unet_md.onnx \
               --FA_soc A2 \
               --TOME_num 10 \
               --faster_gelu \
               --batch_size ${bs} \
               --parallel
         ```
         参数说明：
         - --model：onnx模型路径。
         - --new_model：优化后生成的onnx模型路径。
         - --FA_soc：使用FA算子的硬件形态。目前FlashAttention算子支持Atlas 300I Duo/Pro和Atlas 800I A2，请根据硬件设置参数为Duo或A2，其他不支持硬件请设置为None。
         - --TOME_num：插入TOME插件的数量，有效取值为[0, 10]。如果设置这个参数对精度造成影响，建议调小此值。目前支持Atlas 300I Duo/Pro和Atlas 800I A2，其他不支持硬件请设置为0。默认选取10。
         - --faster_gelu：使用slice+gelu的融合算子。
         - --batch_size：生成适用于指定batch_size的模型，默认值为1。
         - --parallel：生成适用于并行方案的模型

         FA、TOME、Gelu融合算子需通过安装与CANN版本对应的推理引擎包(MindIE-RT)来获取，如未安装推理引擎或使用的版本不支持FA、TOME、SliceGelu算子，FA_soc和TOME_num参数请使用默认配置、不设置faster_gelu参数。

         多batch场景限制：A2场景下暂不支持FA算子优化，FA_soc参数请设置为None。

      3. 适配cache方案(可选，可提升性能但可能导致精度下降)

         运行unet_cache.py脚本
         ```bash
         python3 unet_cache.py --model models/unet/unet_md.onnx --save_dir models/unet/
         ```

   
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```bash
         source /usr/local/Ascend/ascend-toolkit/set_env.sh

         # 如果安装了推理引擎算子包，需配置推理引擎路径
         source /usr/local/Ascend/mindie-rt/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       310P3     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```bash
         # text_encoder
         cd ./models/text_encoder
         atc --framework=5 \
             --model=./text_encoder.onnx \
             --output=./text_encoder \
             --input_format=ND \
             --input_shape="prompt:${bs},77" \
             --log=error \
             --soc_version=Ascend${chip_name}
         atc --framework=5 \
             --model=./text_encoder_2.onnx \
             --output=./text_encoder_2 \
             --input_format=ND \
             --input_shape="prompt:${bs},77" \
             --log=error \
             --soc_version=Ascend${chip_name}
         
         # unet
         cd ../unet/

         # 不使用cache方案
         atc --framework=5 \
             --model=./unet_md.onnx \
             --output=./unet \
             --input_format=NCHW \
             --log=error \
             --optypelist_for_implmode="Gelu,Sigmoid" \
             --op_select_implmode=high_performance \
             --soc_version=Ascend${chip_name}

         # 使用cache方案
         atc --framework=5 \
            --model=./unet_cache.onnx \
            --output=./unet_cache \
            --input_format=NCHW \
            --log=error \
            --optypelist_for_implmode="Gelu,Sigmoid" \
            --op_select_implmode=high_performance \
            --soc_version=Ascend${chip_name}

         atc --framework=5 \
            --model=./unet_skip.onnx \
            --output=./unet_skip \
            --input_format=NCHW \
            --log=error \
            --optypelist_for_implmode="Gelu,Sigmoid" \
            --op_select_implmode=high_performance \
            --soc_version=Ascend${chip_name}

         cd ../../

         # vae
         atc --framework=5 \
             --model=./models/vae/vae.onnx \
             --output=./models/vae/vae \
             --input_format=NCHW \
             --input_shape="latents:${bs},4,128,128" \
             --log=error \
             --soc_version=Ascend${chip_name}

         # 如果使用ddim采样器
         atc --framework=5 \
             --model=./models/ddim/ddim.onnx \
             --output=./models/ddim/ddim \
             --input_format=ND \
             --input_shape="noise_pred:${bs},4,128,128;latents:${bs},4,128,128" \
             --log=error \
             --soc_version=Ascend${chip_name} 
         ```
      
      参数说明：
      - --model：为ONNX模型文件。
      - --output：输出的OM模型。
      - --framework：5代表ONNX模型。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --input_shape: 模型的输入shape信息。


      执行成功后生成om模型列表：  
         ```
         |—— models
                 |—— text_encoder
                        |—— text_encoder.om
                        |—— text_encoder_2.om
                 |—— unet
                        |—— unet.om
                 |—— vae
                        |—— vae.om
                 |—— ddim
                        |—— ddim.om
         ```
       
2. 开始推理验证。
    
    安装绑核工具并根据NUMA亲和性配置任务进程与NUMA node 的映射关系是为了排除cpu的影响。

     安装绑核工具
      ```
      yum install numactl
      ```
      查询卡的NUMA node
      ```
      lspci -vs bus-id
      ```
      bus-id可通过`npu-smi info`获得，查询到NUMA node，在推理命令前加上对应的数字。
      ```
      NUMA node0: 0-23
      NUMA node1: 24-47
      NUMA node2: 48-71
      NUMA node3: 72-95
      ```
      查到NUMA node后，使用`lscpu`获得NUMA node对应的CPU核，推荐绑定其中单核以获得更好的性能。
      ```shell
      NUMA node0: 0-23
      NUMA node1: 24-47
      NUMA node2: 48-71
      NUMA node3: 72-95
      ```

   1. 执行推理脚本。

      ```bash
      # 非并行方案
      numactl -C 72 python3 stable_diffusionxl_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50 \
              --use_cache

      # 并行方案
      numactl -C 72 python3 stable_diffusionxl_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./prompts.txt \
              --device 0,1 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50 \
              --use_cache
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --model_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
      - --use_cache: 在推理过程中使用cache。
      - --cache_steps: 使用cache的迭代次数，迭代次数越多性能越好，但次数过多可能会导致精度下降。取值范围为[1, stpes-1]。
      - --scheduler：采样器。可选None、DDIM、Euler、DPM、EulerAncestral、DPM++SDEKarras。None即为默认scheduler。
      
      执行完成后在`./results`目录下生成推理图片。并在终端显示推理时间，参考如下：

      ```
      [info] infer number: 16; use time: 104.6s; average time: 6.542s
      ```
   

## 精度验证<a name="section741711594518"></a>

   由于生成的图片存在随机性，提供两种精度验证方法：
   1. CLIP-score（文图匹配度量）：评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。使用Parti数据集进行验证。
   2. HPSv2（图片美学度量）：评估生成图片的人类偏好评分，分数的取值范围为[0, 1]，越高越好。使用HPSv2数据集进行验证

   注意，由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

   1. 下载Parti数据集

      ```bash
      wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
      ```

   2. 下载模型权重

      ```bash
      # Clip Score 和 HPSv2 均需使用的权重
      GIT_LFS_SKIP_SMUDGE=1 
      git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
      cd ./CLIP-ViT-H-14-laion2B-s32B-b79K

      # HPSv2权重
      wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt
      ```
      也可手动下载[CLIP权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)
      将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下，手动下载[HPSv2权重](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)放到当前路径


   2. 使用推理脚本生成图片

      ```bash
      # Clip Score
      # 非并行方案
      python3 stable_diffusionxl_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50 \
              --use_cache
              
      # 并行方案
      python3 stable_diffusionxl_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 4 \
              --max_num_prompts 0 \
              --device 0,1 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50 \
              --use_cache

      # HPSv2
      # 非并行方案
      python3 stable_diffusionxl_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file_type hpsv2 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50 \
              --use_cache
              
      # 并行方案
      python3 stable_diffusionxl_ascend_infer.py \
              --model ${model_base} \
              --model_dir ./models \
              --prompt_file_type hpsv2 \
              --max_num_prompts 0 \
              --device 0,1 \
              --save_dir ./results \
              --batch_size ${bs} \
              --steps 50 \
              --use_cache
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --model_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。
      - --num_images_per_prompt: 每个prompt生成的图片数量。
      - --max_num_prompts：限制prompt数量为前X个，0表示不限制。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
      - --use_cache: 在推理过程中使用cache。
      - --cache_steps: 使用cache的迭代次数，迭代次数越多性能越好，但次数过多可能会导致精度下降。

      执行完成后会在`./results`目录下生成推理图片，并且会在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。

   4. 计算精度指标
   
      1. CLIP-score

         ```bash
         python3 clip_score.py \
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
      
      2. HPSv2

         ```bash
         python3 hpsv2_score.py \
               --image_info="image_info.json" \
               --HPSv2_checkpoint="./HPS_v2_compressed.pt" \
               --clip_checkpoint="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
         ```

         参数说明：
         - --image_info: 上一步生成的`image_info.json`文件。
         - --HPSv2_checkpoint: HPSv2模型权重文件路径。
         - --clip_checkpointh: Clip模型权重文件路径。

         执行完成后会在屏幕打印出精度计算结果。
   
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

### StableDiffusionxl

| 硬件形态 | batch size | 迭代次数 | 平均耗时    | 优化方案 | 精度  | 采样器 |
| :------: | :-----: | :----: | :--------: | :--------: | :----: | :----: |
| A2  |  1  |   50  |  4.88s   | 非并行，FA+TOME+faster_gelu，unet_cache | 0.376 | ddim |
| DUO |  1  |   50  |  10.44s   | 并行，FA+TOME+faster_gelu，unet_cache | 0.376 | ddim |

性能测试需要独占npu和cpu
