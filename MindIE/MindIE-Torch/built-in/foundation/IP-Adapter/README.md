# IP-Adapter模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)
  
   - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

在Stable Diffusion研究中，如何有效地将文本提示和图像提示整合到预训练的文生图模型中一直一个挑战。IPAdapter通过引入一个轻量级的适配器模块创新地解决了这个问题，请查看[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)。

- 参考实现：
  ```bash
  # IP-Adapter
  https://github.com/tencent-ailab/IP-Adapter
  ```

- 设备支持：
Atlas 800I A2推理设备：支持的卡数为1
Atlas 300I Duo推理卡：支持的卡数为1

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
- 
  | 配套                                                         | 版本     | 环境准备指导                                                 |
  | ------------------------------------------------------------ |--------| ------------------------------------------------------------ |
  | Python                                                       | 3.10.13 | -                                                            |
  | torch| 2.1.0  | -                                                            |
  | 硬件 | Atlas 300I Duo | - |

请以CANN版本选择对应的固件与驱动版本。

该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码，然后把当前目录下的几个文件移到IP-Adapter工程下

   ```bash
   git clone https://github.com/tencent-ailab/IP-Adapter
   mv attention_processor.patch clip.patch export_ts.py inference.py stable_diffusion_patch.py stable_diffusion_pipeline.py requirements.txt ./IP-Adapter
   ```

2. 按照requirements.txt要求的版本安装相关依赖，避免导出模型失败。

   ```bash
   cd IP-Adapter
   pip3 install -r requirements.txt
   ```

3. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```

4. 代码修改

   执行命令：

   ```bash
   python3 stable_diffusion_patch.py
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息和图片生成图片，无需数据集。
   
## 模型推理<a name="section741711594517"></a>

1. 模型转换。【可选】
   使用Pytorch导出pt模型，然后使用MindIE推理引擎转换为适配昇腾的模型。

   0. 获取权重（可选）

        可提前下载权重，放在任意路径，以避免执行后面步骤时可能会出现下载失败。

        [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
   
        [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   
        git clone https://huggingface.co/h94/IP-Adapter IP-Adapter-weights
   
   1. 导出pt模型并进行编译。(可选)
   
      设置模型名称或路径
      ```bash
      # v1.5
      base_model_path="runwayml/stable-diffusion-v1-5"
      
      # vae
      vae_model_path="stabilityai/sd-vae-ft-mse"
      
      # image_encoder
      image_encoder_path="IP-Adapter-weights/models/image_encoder"
      
      # ip_ckpt
      ip_ckpt="IP-Adapter-weights/models/ip-adapter_sd15.bin"
      ```
   
      执行命令：
   
      ```bash
      # 导出pt模型
      python3 export_ts.py \
      --base_model_path ${base_model_path} \
      --vae_model_path ${vae_model_path} \
      --image_encoder_path ${image_encoder_path} \
      --batch_size 1 \
      --output_dir ./models \
      --device 0 \
      --soc Duo
      ```
      
      参数说明：
      - --base_model_path：SD的模型名称或本地模型目录的路径
      - --vae_model_path：VAE的模型名称或本地模型目录的路径
      - --image_encoder_path：image_encoder的模型名称或本地模型目录的路径
      - --output_dir: 导出的模型输出目录
      - batch_size：目前只支持batch为1
      - --device：使用的NPU芯片，默认是0
      - soc：soc_version。默认为Duo，可支持A2
      


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
      numactl -C 0-23 python3 inference.py \
      --base_model_path ${base_model_path} \
      --vae_model_path ${vae_model_path} \
      --image_encoder_path ${image_encoder_path} \
      --ip_ckpt ${ip_ckpt} \
      --output_dir ./models \
      --device 0 \
      --image_path ./assets/images/woman.png \
      --save_image_path ./test.png \
      --prompt "A girl"
      ```
      
      参数说明：
      - --base_model_path：SD的模型名称或本地模型目录的路径
      - --vae_model_path：VAE的模型名称或本地模型目录的路径
      - --image_encoder_path：image_encoder的模型名称或本地模型目录的路径
      - --ip_ckpt：ipadpter的模型名称或本地模型目录的路径
      - --output_dir: 导出的模型输出目录
      - --device：使用的NPU芯片，默认是0
      - --image_path: 输入的图片路径
      - --save_image_path：输出的图片路径
      - --prompt：文本提示词
      
   


# 模型推理性能<a name="ZH-CN_TOPIC_0000001172201573"></a>

性能参考下列数据。

### IP-Adapter

| 硬件形态 | 迭代次数 | 平均耗时 |
| :------: |:----:|:----:|
| Duo  |  50  | 4.09s |
