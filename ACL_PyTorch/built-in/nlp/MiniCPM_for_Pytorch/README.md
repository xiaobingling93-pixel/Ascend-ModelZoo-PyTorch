# MiniCPM 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

MiniCPM是面壁与清华大学自然语言处理实验室共同开源的系列端侧语言大模型，主体语言模型有MiniCPM-1B 仅有 12亿（1.2B） 和 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量



## 输入输出数据<a name="section540883920406"></a>

 输入数据

  | 输入数据 | 数据类型 | 大小                                   | 数据排布格式 |
  | -------- | -------- |--------------------------------------| ------------ |
  | input_ids    | int64 | 1 x 1                                | ND         |
  | attention_mask    | int64 | 1 x 1                                | ND         |
  | position_ids    | int64 | 1 x 1                                | ND         |
  | past_key_values    | int64 | layers,2,1,n_heads, kv_len, head_dim | ND         |


- 输出数据

  | 输出数据 | 数据类型    | 大小                                                  | 数据排布格式 |
  | -------- |---------|-----------------------------------------------------| ------------ |
  | logits  | FLOAT32 | 1 x vocab_size                                      | ND           |
  | out_key_values  | FLOAT16 | layers,2,1,36,kv_len,head_dim] | ND           |





# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                             | 版本                  | 取包地址环境准备指导                                                                                                      |
  |--------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------|
  | 固件与驱动                          | Ascend HDK 24.1.RC3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)                   |
  | CANN                           | CANN 8.0.RC3        | https://cmc-szv.clouddragon.huawei.com/cmcversion/index/releaseView?deltaId=10860207193326848&isSelect=Software |
  | Python                         | 3.9.19              | -                                                                                                               |
  | PyTorch                        | 2.1.0               | -                                                                                                               |
  | 说明：310B推理卡请以CANN版本选择实际固件与驱动版本。 | \                   | \                                                                                                               |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
    # 获取源码 commitId ：1392d7f 此代码库已经不更新，可以按下面取最新版
    git clone https://gitee.com/yinghuo302/ascend-llm
    cd ascend-llm   
    patch -p1 < minicpm.patch
   ```
   
2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```



## 模型推理<a name="section741711594517"></a>

1. 环境搭建。

   - protoc安装
      
      根据昇腾文档选择合适的protoc,此版本配套使用的protoc版本最低为 1.13.0  
      进入https://github.com/protocolbuffers/protobuf/releases下载对应版本
      ```
      # 安装protoc==1.13.0， 找一空闲目录下载
      tar -zxvf protobuf-all-3.13.0.tar.gz
      cd protobuf-3.13.0
      apt-get update
      apt-get install autoconf automake libtool
      ./autogen.sh 
      ./configure
      make -j4
      make install
      sudo ldconfig
      protoc --version # 查看版本号
      ```

   - 算子编译部署
      ```
      # 将./custom_op/matmul_integer_plugin.cc 拷贝到指定路径
      cd MiniCPM_for_Pytorch
      export ASCEND_PATH=/usr/local/Ascend/ascend-toolkit/latest
      cp custom_op/matmul_integer_plugin.cc $ASCEND_PATH/tools/msopgen/template/custom_operator_sample/DSL/Onnx/framework/onnx_plugin/
      cd $ASCEND_PATH/tools/msopgen/template/custom_operator_sample/DSL/Onnx 
      ```
      打开build.sh，找到下面四个环境变量，解开注释并修改如下：
      ```
      export ASCEND_TENSOR_COMPILER_INCLUDE=/usr/local/Ascend/ascend-toolkit/latest/include
      export TOOLCHAIN_DIR=/usr
      export AICPU_KERNEL_TARGET=cust_aicpu_kernels
      export AICPU_SOC_VERSION=Ascend310B4
      ```
   - 编译运行
      ```
      ./build.sh 
      cd build_out/
      ./custom_opp_ubuntu_aarch64.run
      # 生成文件到customize到默认目录 $ASCEND_PATH/opp/vendors/，删除冗余文件
      cd $ASCEND_PATH/opp/vendors/customize
      rm -rf op_impl/ op_proto/
      ```
   

2. 模型转换(进入export_llama目录)。

   1). 替换模型文件。
      替换当前目录下的modeling_minicpm.py到模型的权重目录下      
  
   2). 量化
      ```bash
         python generate_act_scales.py \
            --model-name <model_name_or_path> \
            --output-path <output_act_scales_file_path> \
            --num-samples <num_samples> \
            --seq-len <sequence_length> \
            --dataset-path <path_to_the_calibration_dataset>
      ```
      其中参数：  
      output_act_scales_file_path： 导出onnx时候需要使用  
      model-name： 模型权重路径，请从官方下载，并转换为FP16格式 https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16  
      num-samples： 样本数，可以用默认值  
      seq-len：sequence长度，可以用默认值  
      dataset-path：验证数据集路径，可以从https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json下载使用  




   3). 导出onnx模型。

       
       python export_llama.py --model ${模型文件路径} --output ${输出onnx文件路径}
       
       
- 参数说明：  
         - model_name: 模型名称  
         - model_type: 模型类型  
         - save_path: 模型权重保存文件夹  
         - act-path: 权重信息，为上一步的输出信息，非量化场景可以不需要


   4). 使用ATC工具将ONNX模型转OM模型。

1. 配置环境变量。

         
          source /usr/local/Ascend/ascend-toolkit/set_env.sh


2. 执行命令查看芯片名称（$\{chip\_name\}）。

         
         npu-smi info

#该设备芯片名为Ascend310P3 （自行替换）  
会显如下：  

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
         
          atc --framework=5 --model=${onnx文件路径}  --output=${输出文件名} --input_format=ND --input_shape="input_ids:1,1;attention_mask:1,1025;position_ids:1,1;past_key_values:52,2,1,8,1024,64" --soc_version=Ascend310B1 --precision_mode=must_keep_origin_dtype
          

- 参数说明：

           - model：为ONNX模型文件。  
           - framework：5代表ONNX模型。  
           - output：输出的OM模型。
           - input\_format：输入数据的格式。
           - input\_shape：输入数据的shape。
           - log：日志级别。
           - soc\_version：处理器型号。
   

           运行成功后生成om后缀的模型文件。

3. 开始推理验证。  
   1). 执行推理前准备工作:  
        A）在端侧设备上如310B1 上安装对应cann，驱动等  
        B）进入inference, 安装相关依赖 pip install -r requirements.txt  

   2). 执行推理:

        
        python main.py --model ${om文件路径}  --hf-dir ${模型文件路径} --engine acl --sampling greedy --cli --kv_size 1024
        

- 参数说明：               
             -   model：om模型路径  
             -   hf-dir：需要tokenizer和模型配置文件，权重不需要   
             -   engine：310B上只能acl  
             -   sampling：greedy/top_p/top_k  
             -   cli：表示在终端运行  
             说明: 上面参数根据实际情况修改

3.数据集精度验证:  
       先下载CEval，BoolQ，GSM8K数据集到inference目录下   ，具体路径为./inference/dataset 
       
         python precision.py --model ${om文件路径} --hf-dir ${模型文件路径} --engine acl --sampling greedy --cli --dataset=BoolQ/CEval/GSM8K
        
         
   
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

4. 调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度（Acc） | 性能(tokens/s) |
| :------: | :--------: | :----: |:-------:|:------------:|
|    310B1      |     1       |    BoolQ    |   70%   |      10      |
