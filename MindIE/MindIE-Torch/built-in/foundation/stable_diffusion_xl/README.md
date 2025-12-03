# 模型推理指导  

## 一、模型简介

SDXL是一种文本到图像的扩散模型，能够在给定任何文本输入的情况下生成照片逼真的图像，参考实现请查看[Stable Diffusion xl blog](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)。

本模型使用的优化手段如下：
- 等价优化：FA、DP并行
- 算法优化：UnetCache

本模型输入输出数据：
  **表 1**  输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | prompt    |  1 x 77 | INT64|  ND|

  **表 2**  输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 3 x 1024 x 1024 | FLOAT32  | NCHW          |

## 二、环境准备

  **表 3**  版本配套表

  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
  | torch| 2.1.0  | -                                                            |

### 2.1 获取安装包
- 支持设备：[Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32) / [Atlas 300I Duo](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=2&model=17)
- 支持卡数：Atlas 800I A2支持的卡数为1或2；Atlas 300I Duo支持的卡数为1，可双芯并行
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

### 2.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 2.4 下载本仓库
```shell
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
```

### 2.5 安装所需依赖
按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
```shell
pip install -r requirements.txt

# 使用hpsv2验证精度，需要按照以下步骤安装hpsv2
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip3 install -e .
```
**注意**：本README中的StableDiffusionxl模型推理方式与torch-npu冲突，需卸载torch-npu包。

## 三、模型权重

权重下载
```bash
# 需要使用 git-lfs (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

## 四、模型推理

### 4.1 代码修改
执行命令：
```bash
# 若环境没有patch工具，请自行安装
python3 stable_diffusion_attention_patch.py
# 若使用unetCache
python3 stable_diffusionxl_unet_patch.py
```

### 4.2 模型转换
使用Pytorch导出pt模型，再使用MindIE推理引擎转换为适配昇腾的模型。

1. 设置权重路径：
```bash
# 执行时下载权重
model_base="stabilityai/stable-diffusion-xl-base-1.0"

# 使用上一步下载的权重
model_base="./stable-diffusion-xl-base-1.0"
```

2. 执行命令：
```bash
# 使用unetCache, 非并行
python3 export_ts.py --model ${model_base} --output_dir ./models --use_cache --batch_size 1 --flag 0 --soc A2 --device 0

# 使用unetCache, 并行
python3 export_ts.py --model ${model_base} --output_dir ./models --use_cache --parallel --batch_size 1 --flag 0 --soc Duo --device 0
```

参数说明：
- --model：模型权重路径
- --output_dir: 存放导出模型的路径
- --use_cache: 【可选】推荐在推理过程中使用unetCache策略
- --parallel: 【可选】导出适用于并行方案的模型, 当前仅带unetCache优化时，支持并行
- --batch_size: 设置batch_size, 默认值为1, 当前最大支持batch_size=2
- --flag：默认为0。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512；2代表动态shape，height的范围为[512, 1024]，width的范围是[512, 1664]。
- --soc：只支持Duo和A2。
- --device：推理设备ID

### 4.3 性能测试
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

3. 执行推理脚本
   ```bash
   # 不使用unetCache策略
   numactl -C 0-23 python3 stable_diffusionxl_pipeline.py \
            --model ${model_base} \
            --prompt_file ./prompts.txt \
            --prompt_file_type plain \
            --device 0 \
            --save_dir ./results \
            --steps 50 \
            --output_dir ./models \
            --flag 0 \
            --height 1024 \
            --width 1024 \
            --batch_size 1
   
   # 使用UnetCache策略
   numactl -C 0-23 python3 stable_diffusionxl_pipeline.py \
            --model ${model_base} \
            --prompt_file ./prompts.txt \
            --prompt_file_type plain \
            --device 0 \
            --save_dir ./results_unetCache \
            --steps 50 \
            --output_dir ./models \
            --use_cache \
            --flag 0 \
            --height 1024 \
            --width 1024 \
            --batch_size 1
   
   # 使用UnetCache策略,同时使用双卡并行策略
   numactl -C 0-23 python3 stable_diffusionxl_pipeline_cache_parallel.py \
            --model ${model_base} \
            --prompt_file ./prompts.txt \
            --prompt_file_type plain \
            --device 0,1 \
            --save_dir ./results_unetCache_parallel \
            --steps 50 \
            --output_dir ./models \
            --use_cache \
            --flag 0 \
            --height 1024 \
            --width 1024 \
            --batch_size 1
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
   - --use_cache: 【可选】推荐在推理过程中使用unetCache策略。
   - --flag：默认为0。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512；2代表动态shape，height的范围为[512, 1024]，width的范围是[512, 1664]。**注意**：请与导出模型时设置的flag保持一致
   - --height：与flag标志位对应的height一致
   - --width：与flag标志位对应的width一致
   
   不带unetCache策略，执行完成后在`./results`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。
   带unetCache策略，执行完成后在`./results_unetCache`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。并在终端显示推理时间。
   带unetCache策略，同时使用双卡并行策略，执行完成后在`./results_unetCache_parallel`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。并在终端显示推理时间。

### 4.4 精度测试
由于生成的图片存在随机性，提供两种精度验证方法：
- CLIP-score（文图匹配度量）：评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。使用Parti数据集进行验证
- HPSv2（图片美学度量）：评估生成图片的人类偏好评分，分数的取值范围为[0, 1]，越高越好。使用HPSv2数据集进行验证

**注意**：由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间

1. 下载Parti数据集
   ```bash
   wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
   ```

2. 下载模型权重
   ```bash
   # Clip Score和HPSv2均需要使用的权重
   # 安装git-lfs
   apt install git-lfs
   git lfs install

   # Clip Score权重
   git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
   
   # HPSv2权重
   wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt --no-check-certificate
   ```
   也可手动下载[权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)
   将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下，手动下载[HPSv2权重](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)放到当前路径

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
            --width 1024 \
            --batch_size 1
   
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
            --width 1024 \
            --batch_size 1
   
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

   不带unetCache策略，执行完成后在`./results_PartiPrompts`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。
   带unetCache策略，执行完成后在`./results_PartiPrompts_unetCache`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。并在终端显示推理时间。
   带unetCache策略，同时使用双卡并行策略，执行完成后在`./results_PartiPrompts_unetCache_parallel`目录下生成推理图片，在当前目录生成一个`image_info.json`文件，记录着图片和prompt的对应关系。并在终端显示推理时间。

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


## 五、+Lora热切换的模型推理【可选】
### Lora热切换功能使用
   1. lora热切使用准备
   
      1. 获取权重
      
         sdxl权重地址：
         ```bash
         https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
         ```
         sdxl lora权重地址：
         ```bash
         https://huggingface.co/latent-consistency/lcm-lora-sdxl
         ```
      2. 代码修改

         基础补丁
   
         ```bash
         python3 stable_diffusion_attention_patch.py
         ```
   
         ```bash
         # 若使用unetCache
         python3 stable_diffusionxl_unet_patch.py
         ```
         
         lora补丁

         ```bash
         python3 stable_diffusionxl_lora_patch.py
         ```
   2. 模型转换
   
      设置基础模型路径以及unet基础权重保存路径：
      ```bash
      # 上一步下载的模型路径
      export model_base="./stable-diffusion-xl-base-1.0"
      # unet基础权重保存路径
      export baselora_path="./baselora"
      ```

      导入环境变量：
      ```bash
      export MINDIE_TORCH_ENABLE_RUNTIME_BUFFER_MUTATION=true
      ```

      执行模型转换：
      ```bash
      #基础模型lora热切特性转换：
      python3 export_ts.py --model ${model_base} --output_dir ./models --batch_size 1 --flag 0 --soc A2 --device 0 --lorahot_support --baselora_path ${baselora_path}
      #unetcahche版模型转换：
      python3 export_ts.py --model ${model_base} --output_dir ./models --use_cache --batch_size 1 --flag 0 --soc A2 --device 0 --lorahot_support --baselora_path ${baselora_path}
      ```
      参数说明：
      - --model 下载的模型权重路径
      - --output_dir 转换后的模型输出路径
      - --batch_size 设置batch_size, 默认值为1, 当前最大支持batch_size=2
      - --flag：默认为0。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512；2代表动态shape，height的范围为[512, 1024]，width的范围是[512, 1664]。
      - --soc：只支持Duo和A2。
      - --device：推理设备ID
      - --lorahot_support：生成模型支持Lora热切换功能
      - --baselora_path：仅指定lorahot_support时生效，代表Unet基础权重的保存路径，用于后续Lora权重热切换
   3. 推理验证

      设置lora权重路径：
      ```bash
      # 第一步下载的lora权重路径
      export newlora_path="./lora_weight"
      ```
      开启cpu高性能模式
      ```bash
      echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
      sysctl -w vm.swappiness=0
      sysctl -w kernel.numa_balancing=0
      ```

      安装绑核工具
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
      
      执行推理
      ```bash
      #基础模型使用lora热切换功能推理
      numactl -C 0-23 python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --prompt_file_type plain \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --batch_size 1 \
              --use_loraHotswitch \
              --lorabase_weight ${baselora_path} \
              --loranew_weight ${newlora_path}
      #Unetcache优化接入后lora热切换功能推理
      numactl -C 0-23 python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --prompt_file_type plain \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --batch_size 1 \
              --use_cache \
              --use_loraHotswitch \
              --lorabase_weight ${baselora_path} \
              --loranew_weight ${newlora_path}
      ```
      参数说明
      - --model：模型权重路径。
      - --output_dir：存放导出模型的目录。
      - --prompt_file：提示词文件。
      - --prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。
      - --save_dir：生成图片的存放目录。
      - --batch_size：模型batch size。
      - --steps：生成图片迭代次数。
      - --device：推理设备ID
      - --use_cache: 推理过程中使用unetCache策略。
      - --flag：默认为0。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512；2代表动态shape，height的范围为[512, 1024]，width的范围是[512, 1664]。**注意**：请与导出模型时设置的flag保持一致
      - --height：与flag标志位对应的height一致
      - --width：与flag标志位对应的width一致
      - --use_loraHotswitch: 代表是否有Lora热切换功能启用
      - --lorabase_weight: 基础的Unet权重存储路径
      - --loranew_weight：第一步下载的lora权重路径
   ### Lora热切换功能精度验证

   通过比较冷切模型与热切模型对于同一prompt的出图余弦相似度来衡量热切方法的精度
   
   1. 模型准备：
      1. 冷切模型准备
      ```bash
      #导入融合lora模型权重环境便令
      export model_new="融合后模型保存路径"
      #执行如下命令进行权重融合
      python3 convert_lora_safetensors_to_diffusers.py --base_model_path ${model_base} \
      --checkpoint_path ${newlora_path} --dump_path ${model_new}
      ```
      参数说明：
      - --base_model_path：基础sdxl模型权重路径
      - --checkpoint_path：下载的lora权重路径
      - --dump_path：权重融合后的模型输出路径

      此后运行如下命令生成新权重的pt模型：
      ```bash
      #不使用unetcache
      python3 export_ts.py --model ${model_new} --output_dir ./models --batch_size 1 --flag 0 --soc A2 --device 0
      #使用unetcache
      python3 export_ts.py --model ${model_new} --output_dir ./models --batch_size 1 --flag 0 --soc A2 --device 0 --use_cache
      ```
      2. 热切模型准备
      
         参照"Lora热切换功能使用"章节进行模型导出
      3. 精度验证用clip网络：
      ```bash
      https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
      ```
   2. 准备精度衡量数据集:

      下载parti数据集：
      ```bash
      wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
      ```
   3. 执行推理：

      不使用Unetcache
      ```bash
      #冷切模型推理：
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_new} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 1 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts_wolorahot \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --batch_size 1 \
              --info_file_save_path ./coldModel.json
      #热切模型推理：
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 1 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts_lorahot \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --batch_size 1
              --use_loraHotswitch \
              --lorabase_weight ${baselora_path} \
              --loranew_weight ${newlora_path} \
              --info_file_save_path ./hotModel.json
      ```
      使用Unetcache
      ```bash
      #冷切模型推理：
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_new} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 1 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts_wolorahot \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --batch_size 1 \
              --use_cache \
              --info_file_save_path ./coldModel.json
      #热切模型推理：
      python3 stable_diffusionxl_pipeline.py \
              --model ${model_base} \
              --prompt_file ./PartiPrompts.tsv \
              --prompt_file_type parti \
              --num_images_per_prompt 1 \
              --max_num_prompts 0 \
              --device 0 \
              --save_dir ./results_PartiPrompts_lorahot \
              --steps 50 \
              --output_dir ./models \
              --flag 0 \
              --height 1024 \
              --width 1024 \
              --batch_size 1
              --use_cache \
              --use_loraHotswitch \
              --lorabase_weight ${baselora_path} \
              --loranew_weight ${newlora_path} \
              --info_file_save_path ./hotModel.json
      ```
      新增参数说明：

      - --info_file_save_path：推理任务完成后图像与promt的对应关系会以json文件存储，此参数指定json文件存储路径与存储名称
   4. 执行精度验证脚本：
         ```bash
         python3 lorahot_score.py \
               --device=cpu \
               --image_info_wo_lorahot = "image_info_wo_lorahot.json" \
               --image_info_lorahot = "image_info_lorahot.json" \
               --model_name="ViT-H-14" \
               --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
         ```

         参数说明：
         - --device：Clip网络推理设备
         - --image_info_wo_lorahot：上一步生成的无离线融合模型推理结果
         - --image_info_lorahot：上一步Lora热切换后模型推理结果
         - --model_name：Clip模型结果
         - --model_weights_path：Clip模型权重

## 六、推理结果参考
### StableDiffusionxl性能 & 精度数据

| 硬件形态  | cpu规格 | batch size | 迭代次数 | 优化手段 | 性能 | 精度  | 采样器 |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| Atlas 800I A2(8*32G) | 64核(arm) |  1  |  50  | with UnetCache, w/o 量化 |  4s   | clip score 0.376 | ddim |
| Atlas 800I A2(8*32G) | 64核(arm) |  1  |  50  | with UnetCache, with 量化 |  3.6s   | clip score 0.371 | ddim |

性能测试需要独占npu和cpu

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。