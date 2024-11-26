# Wav2lip模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>
`Wav2Lip`是一种强大的唇形生成模型，其主要特点是不仅可以生成与音频同步的唇形，而且生成的唇形与原始视频中的人脸表情和头部动作保持一致。`Wav2Lip`模型可以用于各种应用，如视频会议、电影制作和虚拟现实。


- 参考实现：

  ```
  url=https://github.com/Rudrabha/Wav2Lip
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                    | 数据排布格式 |
  | -------- | -------- | ----------------------- | ------------ |
  | input1   | FLOAT32  | batchsize x 1 x 80 x 16 | NCHW         |
  | input2   | FLOAT32  | batchsize x 6 x 96 x 96 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                    | 数据排布格式 |
  | -------- | -------- | ----------------------- | ------------ |
  | output   | FLOAT32  | batchsize x 3 x 48 x 96 | NCHW         |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                            | 版本     | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 23.0.rc1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 7.0      | -                                                                                                     |
  | Python                                                          | 3.7.5    | -                                                                                                     |
  | PyTorch                                                         | 1.8.0    | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \        | \                                                                                                     |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/Rudrabha/Wav2Lip.git
   cd Wav2Lip
   ```

2. 安装依赖。

   ```
   # 注释原有requirements.txt中的opencv-contrib-python>=4.2.0.34
   # 修改原有requirements.txt中的torch==1.8.0
   pip3 install -r requirements.txt
   ```

3. 安装ffmpeg。

   ```
   sudo apt-get install ffmpeg
   ```
4. 将迁移文件放置于Wav2lip文件夹
   ```
    Wav2lip
    ├──...
    ├──wav2lip_pth2onnx.py
    ├──wav2lip_preprocess.py
    └──wav2lip_postprocess.py
   ```
## 准备数据集<a name="section183221994411"></a>
1. 获取测试数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用一段视频和音频作为测试数据，[数据地址](https://pan.baidu.com/s/11xKV1srkKJ7b9atUOGDuow?pwd=mhxw) （提取码: mhxw）如下，视频和音频文件分别存放在./testdata。
   ```
   testdata
   ├── video.mp4        // 视频数据       
   └── audio.mp3        // 音频数据
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行wav2lip_preprocess.py脚本，完成预处理。

   ```
   python3 wav2lip_preprocess.py --video ./testdata/video.mp4 --audio ./testdata/audio.mp3 --data_save_dir ./inputs
   ```
   
   - 参数说明：
   
     --video, 输入视频文件存放地址
         
     --audio, 输入音频文件存放地址

     --save_data_dir, 输出预处理后的bin文件存放路径

    得到的inputs文件夹中包含
    ```
   inputs
   ├── imgs.bin        // 视频数据   
   ├── mels.bin        // 音频数据
   ├── frames.bin      // frames数据
   └── coords.bin      // coords数据
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [wav2lip模型预训练pth权重文件](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)，将获取的权重文件放在当前工作路径下。

   2. 导出onnx文件。

      1. 使用pth2onnx.py脚本。

         运行pth2onnx.py脚本。

         ```
         batch_size=72
         python3 pth2onnx.py --checkpoint_path ./wav2lip.pth --onnx_dir ./ -batch_size ${batch_size}
         ```

         获得wav2lip.onnx文件。

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

      3. 执行ATC命令。

         ```
         atc --model=./wav2lip_bs72.onnx --framework=5 --output=./wav2lip_bs72--input_format=ND --input_shape="input1:72,1,80,16;input2:72,6,96,96" --log=debug  --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成wav2lip_bs72.om模型文件

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。
        ```
        python3 -m ais_bench --model ./wav2lip_bs72.om --input "./inputs/mels.bin,./inputs/imgs.bin" --output ./result --batchsize 72
        ```
        -   参数说明：

             -   model：需要推理om模型的路径。
             -   input：模型需要的输入bin文件夹路径。
             -   output：推理结果输出路径。

        推理后的输出默认在当前目录output的目录下以时间戳命名子目录。
           
      ```
      output
      ├── 2024_02_10-12_00_00  
      |   └── mels_0.bin
      └── 2024_02_10-12_00_00_summary.json        
      ```

   3. 数据后处理。

      调用wav2lip_postprocess.py脚本合成完整视频的文件。

      ```
       python3 wav2lip_postprocess.py --om_pred mels_0.bin --frames ./inputs/frames.bin --coords ./inputs/coords.bin --outfile ./results/result_voice.mp4 --audio ./testdata/audio.mp3
      ```

      - 参数说明：

        - om_pred：为生成推理结果所在路径
        - frames：为frames数据所在路径
        - coords：为coords数据所在路径
        - outfile：为输出的合成视频路径及文件名
        - audio：待输入合成的音频文件
    
    4. 执行纯推理验证性能
        ```
        python3 -m ais_bench --model "wav2lip_bs${batch_size}.om" --loop 100
        ```
        - 参数说明：
           - --model：OM模型文件
           - --loop：纯推理次数


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
   | NPU芯片型号 | Batch Size |  数据集  | throughout性能 |
   | :---------: | :--------: | :------: | :------------: |
   | 300I Pro |     1      | 随机数据 |     752.03     |
   | 300I Pro |     2      | 随机数据 |    1196.68     |
   | 300I Pro |     4      | 随机数据 |    1572.50     |
   | 300I Pro |     8      | 随机数据 |    1927.07     |
   | 300I Pro |     16     | 随机数据 |    1965.25     |
   | 300I Pro |     32     | 随机数据 |    1977.51     |
   | 300I Pro |     64     | 随机数据 |    2022.19     |
   | 300I Pro |    128     | 随机数据 |    1943.19     |