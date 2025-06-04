# MGN-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******
# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

MGN网络是一种多分支深度网络架构的特征识别网络，由一个用于全局特征表示的分支和两个用于局部特征表示的分支组成。将图像均匀地划分为几个条纹，并改变不同局部分支中的特征数量，以获得具有多个粒度的局部特征表示。

- 参考实现：

  ```
  url=https://github.com/GNAYUOHZ/ReID-MGN.git
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型    | 大小     | 数据排布格式 |
  |---------|--------|--------| ------------ |
  | input    | Float16 | batchsize x 3 x 384x 128 | ND     |


- 输出数据

  | 输出数据 | 数据类型               | 大小    | 数据排布格式 |
  |------------------|---------| -------- | ------------ |
  | output  | FLOAT16 | batchsize x 1000 | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | >=1.8.0 | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>
1. 获取本仓源码
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/MGN_for_Pytorch
   ```

1. 获取源码。

   ```
   git clone https://github.com/GNAYUOHZ/ReID-MGN.git ./MGN
   patch -R MGN/data.py < module.patch
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型将[Market1501数据集](https://pan.baidu.com/s/1ntIi2Op?_at_=1624593258681) 的训练集随机划分为训练集和验证集，为复现精度这里采用固定的验证集。
   
2. 数据预处理。

   1.将下载好的数据集移动到./ReID-MGN-master/data目录下
   
   2.执行预处理脚本，生成数据集预处理后的bin文件
   
    ```
    # 首先在要cd到ReID-MGN-master目录下.
    python3  ./postprocess_MGN.py --mode save_bin  --data_path ./data/market1501
    ```

3. 生成数据集信息文件

   1.生成数据集信息文件脚本preprocess_MGN.py

   2.执行生成数据集信息脚本，生成数据集信息文件

   ```
   python ./preprocess_MGN.py bin ./data/market1501/bin_data/q/ ./q_bin.info 384 128
   python ./preprocess_MGN.py bin ./data/market1501/bin_data/g/ ./g_bin.info 384 128
   
   python ./preprocess_MGN.py bin ./data/market1501/bin_data_flip/q/ ./q_bin_flip.info 384 128
   python ./preprocess_MGN.py bin ./data/market1501/bin_data_flip/g/ ./g_bin_flip.info 384 128
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       到以下[链接](https://pan.baidu.com/s/12AkumLX10hLx9vh_SQwdyw)下载预训练模型（提取码：mrl5）

   2. 导出onnx文件。

      1. 使用**pth2onnx.py**导出onnx文件。

         运行**pth2onnx.py**脚本。

         ```
         #将model.pt模型转为market1501.onnx模型，注意，生成onnx模型名(第二个参数)和batch size(第三个参数)根据实际大小设置.
         python3.7 ./pth2onnx.py ./model/model.pt ./model/model_mkt1501_bs1.onnx 1    
         ```
         > **说明：** 
         运行成功后文件夹下生成**model_mkt1501_bs1.onnx**模型文件。
      
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
         atc --framework=5 \
             --model=./model/model_mkt1501_bs1.onnx \
             --input_format=NCHW \
             --input_shape="image:1,3,384,128" \
             --output=mgn_mkt1501_bs1 \
             --log=debug \
             --soc_version=Ascend${chip_name}
         ```
         
      - 参数说明：
         
           --model：ONNX模型文件
         
           --framework：5代表ONNX模型
         
           --output：输出的OM模型
         
        --input_format：输入数据的格式
         
        --input_shape：输入数据的shape
         
        --log：日志级别
         
        --soc_version：处理器型号
           
           --insert_op_conf: aipp预处理算子配置文件
         
        运行成功后在output文件夹下生成**om**模型文件。

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b.  执行推理。

      ```
      python3 -m ais_bench --model_type=vision --device_id=0 --batch_size=1 --om_path=mgn_mkt1501_bs1.om --input_text_path=./q_bin.info --input_width=384 --input_height=128 --output_binary=False --useDvpp=False
      ```
    
      -   参数说明：
    
           -   model：需要推理om模型的路径。
           -   input：模型需要的输入bin文件夹路径。
           -   output：推理结果输出路径。
           -   outfmt：输出数据的格式。
           -   output_dirname:推理结果输出子文件夹。
    	...


   c.  精度验证。

后处理统计mAP精度

调用postprocess_MGN.py脚本的“evaluate_om”模式推理结果与语义分割真值进行比对，可以获得mAP精度数据。

```
python3.7 ./postprocess_MGN.py  --mode evaluate_om --data_path ./data/market1501/ 
```

第一个参数为main函数运行模式，第二个为原始数据目录，第三个为模型所在目录。  
查看输出结果：

```
mAP: 0.9423
```

经过对bs8的om测试，本模型batch8的精度没有差别，精度数据均如上。

   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    
      ```
      python3 -m ais_bench --model ./output/mgn_mkt1501_bs1.om --loop 1000 --batchsize ${bs}
    
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | Batch Size | 数据集      | 精度                    | 性能      |
|-------|------------|----------|-----------------------|---------|
| 300I Pro  | 8         | market1501 | mAP=0.9423 | 1519fps |
