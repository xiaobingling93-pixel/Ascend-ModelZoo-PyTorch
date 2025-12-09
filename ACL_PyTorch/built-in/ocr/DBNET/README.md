# DB模型PyTorch离线推理指导


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

在基于分割的文本检测网络中，最终的二值化map都是使用的固定阈值来获取，并且阈值不同对性能影响较大。而在DB中会对每一个像素点进行自适应二值化，二值化阈值由网络学习得到，彻底将二值化这一步骤加入到网络里一起训练，这样最终的输出图对于阈值就会非常鲁棒。 


- 参考实现：

  ```
  url=https://github.com/MhLiao/DB 
  commit_id=4ac194d0357fd102ac871e37986cb8027ecf094e
  model_name=DB_for_PyTorch
  ```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 25.3.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 8.3.RC1 | -                                                            |
  | Python                                                       | 3.11.10   | -                                                            |
  | PyTorch                                                      | 2.1.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

   ```shell
   #下载原仓代码并切换至目标状态
   git clone https://github.com/MhLiao/DB 
   cd DB
   git reset 4ac194d0357fd102ac871e37986cb8027ecf094e --hard
   
   #统一原仓文件格式
   dos2unix backbones/resnet.py

   git apply ../db.diff

   rsync -av --exclude='DB' ../ ./

   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集

   本模型支持total_text验证集。用户需自行下载数据集并解压到`DB/datasets`路径下，可参考[源码数据集](https://github.com/MhLiao/DB#datasets)。目录结构如下：

   ```
   datasets/total_text/  
   ├── test_gts  
   ├── test_images  
   ├── test_list.txt  
   ├── train_gts  
   └── train_list.txt  
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行db_preprocess.py脚本，完成预处理

   ```shell
   python3 ./db_preprocess.py --image_src_path=./datasets/total_text/test_images --npu_file_path=./prep_dataset
   ```
   
   结果存在 ./prep_dataset 中


## 模型推理<a name="section741711594517"></a>
- 获取权重文件。
   用户需自行下载模型文件并解压在项目路径`${DBNET}`路径下，可参考[DB源码仓](https://github.com/MhLiao/DB#Models)。

- 模型转换。

   - 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

      使用db_pth2onnx.py导出onnx文件

      ```shell
      python3 ./db_pth2onnx.py ./DB/experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume ./totaltext_resnet18
      ```
         
      获得dbnet.onnx文件 
      
   - 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。
   
         ```sh
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
      2. 执行命令查看芯片名称（$\{chip\_name\}）。
   
         ```sh
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
   
         ```sh
         atc --framework=5 \
         --model=./dbnet.onnx \
         --input_format=ND \
         --input_shape=${in_shape} \
         --dynamic_dims=${dynamic_dims} \
         --output=db_bs${bs}  \
         --log=error \ 
         --soc_version=Ascend${chip_name} \
         --precision_mode=force_fp32
         ```
      
         运行成功后生成**db_bs${bs}.om**模型文件。
         
         - 参数说明
              - **--dynamic_dims** 动态维度，需要通过`python get_dynamic_dims.py`命令获得打屏信息，随后设置${dynamic_dims}为打屏的数据。
           
              - --model：为ONNX模型文件。
              
              - --framework：5代表ONNX模型。
              
              - --output：输出的OM模型。
              
              - --input_format：输入数据的格式。
              
              - --input_shape：输入数据的shape, 格式严格为`"input_name:n,c,h,w"`。
                              input_name：本模型固定为actual_input_1；
                              n：batchsize，此处为${bs}；
                              h: 高度，本模型固定为800；
                              w：宽度，自适应，为-1；

              - --log：日志级别。--soc\_version：处理器型号。
              
              - --bs：一次推理的样本数，自行设定。

              
                
   
2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```shell
      python3 om_infer.py --device 0 --batchsize 1 --preped_path ./prep_dataset  --output_path ./outputs
      ```

      -   参数说明：
         - --device：int类型，chip id，缺省为0
         - --batchsize: int类型，一次推理的样本数，需与atc导出om模型时设置一致。
         - --preped_path：str类型，经过预处理后的数据集目录，缺省为./prep_dataset
         - --output_path：str类型，推理二进制结果保存的路径，缺省为./outputs

        推理后的结果打屏显示。


   3. 精度验证。

      ```shell
      python3 ./db_postprocess.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --bin_data_path ./outputs --box_thresh 0.7
      ```
      - 参数说明：
        - ./result：om推理结果保存的文件夹
        - result_bs1.json：为精度生成结果文件

   4. 多卡推理demo。
      ```shell
      python3 multi_infer.py --device 0,1 --batchsize 1 --preped_path ./prep_dataset
      ```
      -   参数说明：
         - --device：str类型，chip ids，缺省为0,1
         - --batchsize: int类型，填一次推理的样本数，需与atc导出om模型时设置一致。
         - --preped_path：str类型，经过预处理后的数据集目录，缺省为./prep_dataset


        推理后的结果打屏显示。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

| 芯片型号 | Batch Size |  数据集   | precision:精度 | 性能 (samples/s)  |
| :------: | :--------: | :-------: | :--: | :---: |
|  300I DUO   |     1      | totaltext | 0.88 | 36.37 |
|  300I DUO   |     4      | totaltext |  | 24.82 |
|  300I DUO   |     8      | totaltext |  | 20.60 |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
