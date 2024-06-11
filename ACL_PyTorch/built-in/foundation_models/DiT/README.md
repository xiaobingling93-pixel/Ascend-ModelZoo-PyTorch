# DiT模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DiT模型，即Diffusion Transformers，是一种用于扩散模型的新架构。它的设计目标是尽可能忠实于标准Transformer架构，以保留其可扩展性。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/DiT
  commit_id=ed81ce2229091fd4ecc9a223645f95cf379d582b
  model_name=DiT-XL/2
  ```


## 输入输出数据<a name="section540883920406"></a>

image_num为需要生成的图片数量

batch_size = image_num * 2

latent_size = image_size // 8

- 输入数据

  | 输入数据 | 数据类型  | 大小                                        | 数据排布格式 |
  | -------- | -------- | ------------------------------------------ | ------------ |
  | input    | FLOAT32  | batch_size x 4 x latent_size x latent_size | NCHW         |
  | t        | INT64    | batch_size                                 | ND           |
  | input.5  | INT64    | batch_size                                 | ND           |


- 输出数据

  | 输出数据  | 数据类型 | 大小                                       | 数据排布格式 |
  | -------- | -------- | ----------------------------------------- | ------------ |
  | output   | FLOAT32  | image_num x 4 x latent_size x latent_size | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动
  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                   |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                    | 23.0.0  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 7.0.0 | -                                                              |
  | Python                                                       | 3.10.0   | -                                                           |
  | PyTorch                                                      | 2.1.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。  | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/facebookresearch/DiT
   cd DiT
   git reset --hard ed81ce2229091fd4ecc9a223645f95cf379d582b
   cd ../
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

3. 安装mindie包

   ```bash
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入图片类别信息生成图片，无需数据集。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pt转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      DiT权重文件下载链接如下，按需下载：
       
      [DiT-XL-2-256x256下载链接](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)
       
      [DiT-XL-2-512x512下载链接](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt)

      VAE(变分自编码器)是一种神经网络模型的一部分,它可以对图像进行编码和解码,将图像转换到更小的潜在空间,以便计算可以更快。它通过编码和解码图像来实现在更小的潜在空间中的图像表示,这可以加速计算过程。Stability AI发布了两种精调后的VAE解码器变体,ft-EMA和ft-MSE，它们强调的部分不同。EMA和MSE：指数移动平均（Exponential Moving Average）和均方误差（Mean Square Error）是测量自动编码器好坏的指标。
      
      获取VAE权重文件，按需下载：

      [mse下载链接](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)

      [ema下载链接](https://huggingface.co/stabilityai/sd-vae-ft-ema/tree/main)

      按如下目录结构进行部署：

      ```
       ├── DiT
       |   ├── diffusion
       |   ├── models.py
       |   ├── ......
       ├── stabilityai
       |   ├── sd-vae-ft-ema
       |   |    ├── diffusion_pytorch_model.bin
       |   |    ├── diffusion_pytorch_model.savetensors
       |   |    ├── config.json
       |   |    ├── README.md
       |   ├── sd-vae-ft-mse
       ├── sample_npu.py
       ├── sample_ddp_npu.py
       ├── ......
      ```

   2. 导出onnx文件。

      1. 移动DiT_pt2onnx.py至DiT目录，使用DiT_pt2onnx.py导出onnx文件。

         ```
         mv DiT_pt2onnx.py ./DiT/
         python3 ./DiT/DiT_pt2onnx.py --image_size 512 --model_path ./DiT-XL-2-512x512.pt
         ```

         运行成功后生成<u>***dit_dynamic_512.onnx***</u>和<u>***vae_dynamic_512_mse.onnx***</u>模型文件。

         - 参数说明：

           -   --model\_path: pt文件路径。
           -   --save\_dir: 生成的onnx保存路径。
           -   --image\_size: 输入图片的大小。
           -   --model\_name: 模型名称。
           -   --num\_classes: 类别数量。
           -   --vae: 图片解码模式。

      2. 使用onnxsim精简onnx文件

         ```
         python3 -m onnxsim ./models/dit_onnx/dit_dynamic_512.onnx ./models/dit_onnx/dit_dynamic_512_sim.onnx
         ```

         运行成功后生成<u>***dit_dynamic_512_sim.onnx***</u>模型文件。

         ```
         # duo卡环境执行
         batch_size=1
         python3 opt_dit.py \
            --model ./models/dit_onnx/dit_dynamic_512_sim.onnx \
            --output ./models/dit_onnx/dit_bs${batch_size}_512_opt.onnx \
            --batch_size ${batch_size} \
            --FA_soc Duo
         ```
         - 参数说明：

           -   --model：输入的ONNX模型文件。
           -   --output：输出的ONNX文件路径。
           -   --batch\_size：batchsize大小。
           -   --FA\_soc：处理器型号。

         运行成功后生成<u>***dit_bs${batch_size}_512_opt.onnx***</u>模型文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

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

      3. 执行ATC命令转换DiT模型的onnx文件。

         ```
         # Duo卡
         batch_size=1
         atc --model=./models/dit_onnx/dit_bs${batch_size}_512_opt.onnx \
             --framework=5 \
             --output=./models/om/dit_bs${batch_size}_512_opt \
             --input_format=NCHW \
             --input_shape="input:${batch_size},4,64,64;t:${batch_size};input.5:${batch_size}" \
             --log=error \
             --soc_version=Ascend${chip_name}
         
         # A2
         batch_size=2
         atc --model=./models/dit_onnx/dit_dynamic_512_sim.onnx \
             --framework=5 \
             --output=./models/om/dit_bs${batch_size}_512_sim \
             --input_format=NCHW \
             --input_shape="input:${batch_size},4,64,64;t:${batch_size};input.5:${batch_size}" \
             --log=error \
             --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***dit_bs${batch_size}_512.om***</u>模型文件。

      4. 执行ATC命令转换vae模型的onnx文件。

         ```
         vae_bs=1
         atc --model=./models/vae_onnx/vae_dynamic_512_mse.onnx \
             --framework=5 \
             --output=./models/om/vae_bs${vae_bs}_512_mse \
             --input_format=NCHW \
             --input_shape="latents:${vae_bs},4,64,64" \
             --log=error \
             --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***vae_bs${vae_bs}_512_mse.om***</u>模型文件。


2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 开启cpu高性能模式

      ```bash
      echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
      sysctl -w vm.swappiness=0
      sysctl -w kernel.numa_balancing=0
      ```

   3. 执行推理。

      移动sample_npu.py, background_session.py至DiT目录，进行模型推理。

      ```
      mv ./sample_npu.py ./DiT/
      mv ./background_session.py ./DiT/

      # Duo
      python3 ./DiT/sample_npu.py \
         --image_size 512 \
         --model ./models/om/dit_bs${batch_size}_512_opt.om \
         --vae ./models/om/vae_bs${vae_bs}_512_mse.om \
         --parallel

      # A2
      python3 ./DiT/sample_npu.py \
         --image_size 512 \
         --model ./models/om/dit_bs${batch_size}_512_sim.om \
         --vae ./models/om/vae_bs${vae_bs}_512_mse.om
      ```

      - 参数说明：

        -   --model: dit模型的om文件路径。
        -   --vae: vae模型的om文件路径。
        -   --image\_size: 输入图片大小。
        -   --num\_classes: 图片类别数量。
        -   --num\_sampling\_steps: 每批次输入做推理的次数。
        -   --device\_id： NPU设备编号。
        -   --parallel: 使用并行模式。
        -   --class\_label: 生成图片的类别，输入-1测量全量类别。
        -   --results：生成的图片存放路径

        执行完成后再当前目录生成推理图片sample_npu.png，每次测试结果不同。

   3. 精度验证。

      下载数据集[ImageNet512x512](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz)，放在任意路径

      然后执行以下命令：

      ```
      # Duo
      python3 ./DiT/sample_npu.py \
         --image_size 512 \
         --model ./models/om/dit_bs${batch_size}_512_opt.om \
         --vae ./models/om/vae_bs${vae_bs}_512_mse.om \
         --parallel \
         --class_label -1

      # A2
      python3 ./DiT/sample_npu.py \
         --image_size 512 \
         --model ./models/om/dit_bs${batch_size}_512_sim.om \
         --vae ./models/om/vae_bs${vae_bs}_512_mse.om \
         --class_label -1
      ```

      之后进行FID计算：

      ```
      python3 -m pytorch_fid ./VIRTUAL_imagenet512.npz ./results 
      ```

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=./models/om/dit_bs${batch_size}_512.om --loop=100 --batchsize=${batch_size} --device 0
      ```

      - 参数说明：

        -   --model：om文件路径。
        -   --batchsize：batch大小。
        -   --loop: 纯推理次数。
        -   --device: NPU设备编号。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 硬件形态 | 迭代次数 | 平均耗时 |
| -------- | -------- | -------- |
| Duo      | 250      | 22.17s    |
| A2       | 250      | 10.79s   |