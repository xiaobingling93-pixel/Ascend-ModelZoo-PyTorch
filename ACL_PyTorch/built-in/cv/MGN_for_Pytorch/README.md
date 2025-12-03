# MGN-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [获取源码](#获取源码)
- [准备数据集](#准备数据集)
- [模型转换](#模型转换)
- [文件目录结构](#文件目录结构)
- [模型推理](#模型推理)
- [推理结果验证](#推理结果验证)

# 概述

MGN网络是一种多分支深度网络架构的特征识别网络，由一个用于全局特征表示的分支和两个用于局部特征表示的分支组成。将图像均匀地划分为几个条纹，并改变不同局部分支中的特征数量，以获得具有多个粒度的局部特征表示。

- 参考实现：

  ```
  url=https://github.com/GNAYUOHZ/ReID-MGN.git
  ```

## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型    | 大小     | 数据排布格式 |
  |---------|--------|--------| ------------ |
  | input    | Float16 | batchsize x 3 x 384x 128 | ND     |


- 输出数据

  | 输出数据 | 数据类型               | 大小    | 数据排布格式 |
  |------------------|---------| -------- | ------------ |
  | output  | FLOAT16 | batchsize x 1000 | ND           |



# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本         | 环境准备指导                                                 |
| ------------------------------------------------------------ |------------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 25.0.rc1.1 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 8.1.RC1    | -                                                            |
| Python                                                       | 3.9        | -                                                            |
| PyTorch                                                      | >=1.8.1    | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \          | \                                                            |


# 获取源码
1. 获取本仓源码
   
   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/MGN_for_Pytorch
   ```

2. 获取源码并执行diff文件。

   ```
   git clone https://github.com/GNAYUOHZ/ReID-MGN.git ./MGN
   cd MGN
   git apply ../module.patch
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

# 准备数据集

1. 获取原始数据集。

   该模型将[Market1501数据集](https://pan.baidu.com/s/1ntIi2Op?_at_=1624593258681) 的训练集随机划分为训练集和验证集，为复现精度这里采用固定的验证集。
   
2. 数据预处理。

   1.将下载好的数据集`zip`文件解压，生成`Market-1501-v15.09.15`文件夹
   
   2.执行数据预处理脚本，生成数据集预处理后的bin文件
   
    ```
    # 1. 配置MGN外部路径
    export PYTHONPATH="MGN:$PYTHONPATH"
    # 2. 执行数据预处理
    python3 mgn_preprocess.py --data_path=./Market-1501-v15.09.15
    ```
   参数说明：
   * `--data_path`是数据集的路径，默认值为`./Market-1501-v15.09.15`
   
   如果你的配置参与与默认值相同，可以不用指定，直接执行`python3 mgn_preprocess.py`。

# 模型转换
1. 下载原始模型权重：到以下[链接](https://pan.baidu.com/s/12AkumLX10hLx9vh_SQwdyw)下载预训练模型（提取码：mrl5）
2. 原始`pt`模型转`onnx`模型：
    ```
    # 1.配置MGN外部路径
    export PYTHONPATH="MGN:$PYTHONPATH"
    # 2. 执行模型转换脚本
    python3 mgn_convert.py --model_path=./model --model_weight_file=model.pt --onnx_file=model_mkt1501_bs1.onnx --batchonnx=1
    ```
   参数说明：
   * `--model_path`是模型权重文件的路径，默认值是`./model`
   * `--model_weight_file`是模型权重文件名，默认值是`model.pt`
   * `--onnx_file`是生成的onnx名称，默认值是`model_mkt1501_bs1.onnx`
   * `--batchonnx`是生成onnx时配置的batch_size，默认值是`1`
   
    如果你的配置参与与默认值相同，可以不用指定，直接执行`python3 mgn_convert.py`。
3. `onnx`模型转`om`模型
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
               --input_shape="image:1,3,384,128" \  # 这里的1是指batch_size
               --output=mgn_mkt1501_bs1 \
               --log=debug \
               --soc_version=Ascend${chip_name}
            ```
         
      - 参数说明：

           --framework：指定输入模型的框架类型, 5代表ONNX模型
   
           --model：ONNX模型文件
         
           --output：输出的OM模型
         
           --input_format：输入数据的格式
         
           --input_shape：输入数据的shape
         
           --log：日志级别
         
           --soc_version：昇腾处理器型号
         
        运行成功后在当前文件夹下生成**mgn_mkt1501_bs1.om**模型文件。

# 文件目录结构

准备工作完成后，文件目录结构大致如下：

    ```text
    📁 MGN_for_Pytorch/
    ├── 📁 MGN/  # MGN源码
    ├── 📁 model/  # MGN模型权重
    |   |── 📄 model.pt
    |   └── 📄 model_mk1501_bs1.onnx
    ├── 📁 Market-1501-v15.09.15/  # 数据集
    |   |── 📁 bin_data
    |   |   |── 📁 q     # query数据原始文件
    |   |   └── 📁 g     # gallery数据原始文件
    |   └── 📁 bin_data_flip
    |       |── 📁 q     # query数据flip文件
    |       └── 📁 g     # gallery数据flip文件
    │── 📄 mgn_convert.py 
    │── 📄 mgn_evaluate.py
    │── 📄 mgn_infer.sh
    │── 📄 mgn_preprocess.py
    │── 📄 module.patch
    └── 📄 om_executor.py 
    ```
    
# 模型推理

1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

2. 执行推理。
    ```
   chmod +x mgn_infer.sh
   ./mgn_infer.sh
   ```
   `mgn_infer.sh`内包含4条`python`推理命令，分别是`query`数据、`gallery`数据、`query_flip`数据、`gallery_flip`数据。

   `mgn_infer.sh`脚本支持自定义传入参数，例如：
   ```
   ./mgn_infer.sh -m "mgn_mkt1501_bs1.om" -b 1 -i "./Market-1501-v15.09.15" -o "./result" -f "TXT"
   ```
   -   参数说明：
        -   -m：需要推理om模型的路径。
        -   -b：batch size的大小。
        -   -i：输入数据的路径。
        -   -o: 推理结果输出文件夹路径。
        -   -f：输出数据的格式。
   

# 推理结果验证

1. 精度验证

    后处理统计mAP精度

    使用mgn_evaluate.py，对推理结果与语义分割真值进行比对，可以获得mAP精度数据。

    ```
    # 1.配置MGN外部路径
    export PYTHONPATH="MGN:$PYTHONPATH"
    # 2. 执行结果验证脚本
    python3 mgn_evaluate.py  --result=./result
    ```
    参数说明：
    * `--result`是推理输出结果的路径，默认值为`./result`
   
    如果你的配置参与与默认值相同，可以不用指定，直接执行`python3 mgn_evaluate.py`。


2. 性能验证。

    可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    
    ```
    export bs=1
    python3 -m ais_bench --model=mgn_mkt1501_bs1.om --loop=1000 --batchsize=${bs}
    ```

## 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | Batch Size | 数据集      | 精度                    | 性能      |
|-------|------------|----------|-----------------------|---------|
| 300I Pro  | 8          | market1501 | mAP=0.9423 | 1519fps |
