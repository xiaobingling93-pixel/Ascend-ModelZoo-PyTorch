# Juggernaut-XI-Lightning模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)
  
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>
Juggernaut_XI_Lightning是stable-diffusion-xl的finetune版本
参考实现：
```bash
# Juggernaut-XI-Lightning
https://huggingface.co/Rundiffusion/Juggernaut-XI-Lightning
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小      | 数据类型                | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | prompt    |  batch x 77 | STRING |  ND|


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batch x 3 x 1024 x 1024 | FLOAT32  | NCHW          |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.x | -                                                            |
   | torch| 2.1.0  | -                                                            |

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

0. 下载仓库到本地。
   ```bash
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   ```

1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt

   # 若要使用hpsv2验证精度，则还需要按照以下步骤安装hpsv2
   git clone https://github.com/tgxs002/HPSv2.git
   cd HPSv2
   pip3 install -e .
   ```

2. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息生成图片，无需数据集。


## 模型推理<a name="section741711594517"></a>

1. 获取权重（可选）

    可提前下载权重，放到代码同级目录下，以避免执行后面步骤时可能会出现下载失败。

    ```bash
    # 需要使用 git-lfs (https://git-lfs.com)
    apt install git-lfs
    git lfs install
    
    # Juggernaut-XI-Lightning
    git clone https://huggingface.co/RunDiffusion/Juggernaut-XI-Lightning
    ```

2. 开始推理验证。
                   
    1. 开启cpu高性能模式
    ```bash
    echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    ```

    2. 设置调度流水优化
    ```bash
    export TASK_QUEUE_ENABLE=2
    ```

    3. 执行推理脚本。
    ```bash
    # 使用上一步下载的权重
    model_base="./Juggernaut-XI-Lightning"
    ```

   单卡推理
    ```bash
    python3 ./inference.py \
    --path=${model_base} \
    --prompt_file="./prompts.txt" \
    --height=1024 \
    --width=1024 \
    --steps=12 \
    --num_images_per_prompt=1 \
    --output_dir="./results" \
    --device_id=0 \
    --cache_method="agb_cache"
   ```

    参数说明:
      - --path：模型权重路径
      - --prompt_file: 输入的prompt文件
      - --height: 生成图片的高
      - --width: 生成图片的宽
      - --steps: 推理步数
      - --num_images_per_prompt: 单个prompt生成的图片数量
      - --output_dir: 生成图片的保存路径
      - --iterator: 测试循环次数,取从第3次开始后续的推理耗时的均值
      - --device_id：推理设备ID
      - --cache_method: cache策略选择,支持配置"agb_cache"
   注：性能测试取从第3次后的推理耗时的均值
   
   双卡推理
   ```bash
    ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --master_port=2025 --nproc_per_node=2  ./inference.py \
    --path=${model_base} \
    --prompt_file="./prompts.txt" \
    --height=1024 \
    --width=1024 \
    --steps=12 \
    --num_images_per_prompt=1 \
    --output_dir="./results" \
    --device_id=0 \
    --cache_method="agb_cache" \
    --use_parallel
    ```
    参数说明:
      - --master_port: master节点的端口号，同于通信
      - --nproc_per_node: 一个节点中显卡的数量
      - --use_parallel: 开启双卡并行推理

    
3. 参考性能结果:
   | 800I A2 32G | 分辨率 | 迭代次数 | 单卡推理 | 双卡推理|
   |-------|------|------|--------|-----------|
   |无损优化| 1024x1024 | 12 | 2.09s | 1.99s |
   |算法优化| 1024x1024 | 12 | 1.67s | 1.44s |
   

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
      python3 ./test_precision.py \
      --path=${model_base} \
      --device_id=0 \
      --height=1024 \
      --width=1024 \
      --num_images_per_prompt=4 \
      --cache_method="agb_cache" \
      --prompt_file=./PartiPrompts.tsv \
      --prompt_file_type=parti
      ```

      参数说明：
      - --path：模型权重路径。
      - --height: 生成图片的高
      - --width: 生成图片的宽
      - --num_images_per_prompt: 每个prompt生成的图片数量。注意使用hpsv2时，设置num_images_per_prompt=1即可。
      - --device_id：推理设备ID。
      - --cache_method: cache策略选择,默认为空,代表不开启cache,支持配置"static_cache"和"agb_cache","abg_cache"性能更好,但会引入更大的内存占用,当前"agb_cache"只支持50步迭代场景。
      - --prompt_file：提示词文件。
      - --prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2

      运行完成后的结果会保存到以运行时间戳命名的文件夹下,文件夹下包含结果'image_info.json'文件以及保存的图片文件夹。

   4. 计算精度指标
   
      1. CLIP-score

         ```bash
         python3 clip_score.py \
               --image_info="image_info.json" \
               --model_name="ViT-H-14" \
               --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
         ```

         参数说明：
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

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。