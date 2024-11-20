# stable-video-diffusion模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)
  
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   stable-video-diffusion是一种图像到视频的扩散模型，能够在给定任何图像输入的情况下生成与图像相对应的视频。有关稳定扩散函数的更多信息，请查看[Stable Video Diffusion blog](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)。

- 参考实现：
  ```bash
   # StableVideoDiffusion
   https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
  ```

- 设备支持：
Atlas 800I A2推理设备：支持的卡数为1或2

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | input    |  1 x 512 x 512 x 3 | FLOAT32 |  NHWC |


- 输出数据

  | 输出数据 | 大小      | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output   | 1 x 25 x 512 x 512 x 3 | FLOAT32  | NTHWC |

**注意**：该模型当前仅支持batch size为1的情况。

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
- 
  | 配套                                                         | 版本     | 备注                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
  | torch | 2.0.0  | 导出pt模型所需版本                                            |
  | torch | 2.1.0  | 模型编译和推理所需版本                                         |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。
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
   python3 stable_video_diffusion_activations_patch.py
   ```

   ```bash
   python3 stable_video_diffusion_attention_patch.py
   ```

   ```bash
   python3 stable_video_diffusion_transformer_patch.py
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入图像示例的下载网址为：https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png
   用户自网址自行下载后放置当前路径下，命名为 rocket.png

## 模型推理<a name="section741711594517"></a>

1. 模型转换。
   使用Pytorch导出pt模型，然后使用MindIE推理引擎转换为适配昇腾的模型。

   1. 获取权重

       可提前下载权重，以避免执行后面步骤时可能会出现下载失败。

       ```bash
       # 需要使用 git-lfs (https://git-lfs.com)
       git lfs install
       
       # StableVideoDiffusion
       git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
       ```

   2. 导出pt模型

      设置模型名称或路径
      ```bash
      # 执行时下载权重
      model_base="stabilityai/stable-video-diffusion-img2vid-xt"
      
      # 使用上一步下载的权重
      model_base="./stable-video-diffusion-img2vid-xt"
      ```

      执行命令：
   
      ```bash
      # 导出pt模型
      python3 export_ts.py --model ${model_base} --output_dir ./models
      # 更换torch版本，执行后续的模型编译和推理
      python3 uninstall torch
      python3 install torch==2.1.0
      ```

      参数说明：
      - --model：模型名称或本地模型目录的路径
      - --output_dir: pt模型输出目录

      执行成功后会生成pt模型:
         - ./models/image_encoder_embed/image_encoder_embed.pt
         - ./models/unet/unet_bs2.pt
         - ./models/vae/vae_encode.pt
         - ./models/vae/vae_decode.pt

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
      # 0.第一次推理需要配置环境变量，使得在静态TLS块中可以分配内存：
      find / -name *libGL* # 查找libGLdispatch.so.0文件的路径，记为lib_dir，例如 lib_dir="/lib/aarch64-linux-gnu"
      export LD_PRELOAD=${lib_dir}/libGLdispatch.so.0:$LD_PRELOAD

      # 1.若不使用并行推理：
      numactl -C 0-23 python3 stable_video_diffusion_pipeline.py \
              --model ${model_base} \
              --img_file ./rocket.png \
              --device 0 \
              --save_dir ./results \
              --num_inference_steps 25 \
              --output_dir ./models

      # 2.若使用并行推理：
      numactl -C 0-23 python3 stable_video_diffusion_pipeline_parallel.py \
              --model ${model_base} \
              --img_file ./rocket.png \
              --device 0,1 \
              --save_dir ./results \
              --num_inference_steps 25 \
              --output_dir ./models
      ```
      
      参数说明：
      - --model：模型名称或本地模型目录的路径。
      - --img_file：输入图像文件。
      - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。
      - --save_dir：生成视频的存放目录。
      - --num_inference_steps：生成视频的迭代次数。
      - --output_dir: 编译好的模型路径。
      
      执行完成后在`./results`目录下生成推理视频。并在终端显示推理时间。

   **注意**：若使用800I A2单卡推理，则需要保证单卡的实际可用内存（HBM-Usage最大值-无进程时初始值）> 29762MB。否则尝试重启服务器以降低无进程时初始值、更换服务器，或使用双卡并行推理。


# 模型推理性能<a name="ZH-CN_TOPIC_0000001172201573"></a>

性能参考下列数据。

### StableVideoDiffusion

| 硬件形态 | 迭代次数 | 平均耗时 |
| :------: |:----:|:----:|
| 单卡  |  25  |  28s    |
| 双卡  |  25  |  14.5s  |

**注意**：当前推理pipline中未固定随机种子

```bash
# 推理pipline main函数中加入
generator = torch.Generator().manual_seed(xxx)
# 在ascendie_infer函数中加入参数
generator=generator
```