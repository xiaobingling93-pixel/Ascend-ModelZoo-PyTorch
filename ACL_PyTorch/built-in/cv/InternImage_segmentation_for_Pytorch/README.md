# InternImage Segmentation 动态推理指导

- [概述](#summary)
  
  - [输入数据](#input_data)

- [推理环境准备](#env_setup)

- [快速上手](#quick_start)
  
  - [获取源码](#get_code)
  
  - [下载数据集](#download_data)
  
  - [模型推理](#infer)

- [模型推理性能 & 精度](#performance)

# 概述<a id="summary"></a>

InternImage 是一个由上海人工智能实验室、清华大学等机构的研究人员提出的基于卷积神经网络（CNN）的视觉基础模型。与基于 Transformer 的网络不同，InternImage 以可变形卷积 DCNv3 作为核心算子，使模型不仅具有检测和分割等下游任务所需的动态有效感受野，而且能够进行自适应的空间聚合。在 16 个重要的视觉基础数据集（覆盖分类、检测和分割任务）上取得世界最好性能。此指导仅针对InternImage项目下，backbone为InternImage-H，method为UperNet，分辨率为896x896的模型。本项目以mIoU作为精度评估指标

- 版本说明:
  
  ```
  url=https://github.com/OpenGVLab/InternImage
  commit_id=31c962dc6c1ceb23e580772f7daaa6944694fbe6
  model_name=InternImage
  ```

## 输入数据<a id="input_data"></a>

InternImage使用公共数据集ADE20K进行推理

| 输入数据 | 数据类型     | 大小          | 数据排布格式 |
|:----:|:--------:|:-----------:|:------:|
| img  | RGB_FP32 | （1，3，-1，-1） | NCHW   |

# 

# 推理环境准备<a id="env_setup"></a>

该模型需要以下依赖

表1 **版本配套表**

| 依赖      | 版本      | 环境准备指导                                                                                                        |
| ------- | ------- |:-------------------------------------------------------------------------------------------------------------:|
| 固件与驱动   | 24.1.0  | [PyTorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies/pies_00001.html) |
| PyTorch | 2.1.0   | -                                                                                                             |
| CANN    | 8.0.RC2 | -                                                                                                             |
| Python  | 3.9     | -                                                                                                             |

# 快速上手<a id="quick_start"></a>

## 获取源码<a id="get_code"></a>

1. 获取本仓源码
   
   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/InternImage_segmentation_for_Pytorch
   ```

2. 获取模型仓**InternImage**源码和依赖仓**mmsegmentation**源码
   
   ```git
   git clone -b v1.2.2 https://github.com/open-mmlab/mmsegmentation.git
   git clone https://github.com/OpenGVLab/InternImage.git
   cd InternImage
   git reset --hard 31c962dc6c1ceb23e580772f7daaa6944694fbe6
   cd ../mmsegmentation/
   git reset --hard c685fe6767c4cadf6b051983ca6208f1b9d1ccb8
   cd ..
   ```

3. 转移文件夹位置
   
   ```
   mv internimage_segmentation.patch InternImage/segmentation/
   mv *.py InternImage/segmentation/
   mv exceptionlist.cfg InternImage/segmentation/
   mv mmseg.patch mmsegmentation/mmseg/
   ```

4. 安装依赖
   
   ```
   pip3 install -r requirement.txt
   ```

5. 更换当前路径并打补丁，修改完mmseg源码后进行安装
   
   ```
   cd mmsegmentation/mmseg/
   patch -p2 < mmseg.patch
   pip install -v -e ..
   
   cd ../../InternImage/segmentation/
   patch -p2 < internimage_segmentation.patch
   ```

## 下载数据集<a id="download_data"></a>

    使用下面的链接下载数据集并解压放在InternImage/segmentation/data目录下

> [ADE20K数据集下载](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)

    确保data下的路径结构如下

```
├── ADEChallengeData2016
│   ├── annotations
│   │   ├── training
│   │   ├── validation
│   ├── images
│   │   ├── training
│   │   ├── validation
```

## 模型推理<a id="infer"></a>

1. 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件
   
   1. 下载权重文件并放到InternImage/segmentation/ckpt下
      
      > [ckpt文件下载](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_h_896_160k_ade20k.pth)
   
   2. 导出onnx文件
      
      确认当前路径为InternImage/segmentation并执行如下命令导出onnx文件
      
      ```
      python3 export2onnx.py --config configs/ade20k/upernet_internimage_h_896_160k_ade20k.py --ckpt ckpt/upernet_internimage_h_896_160k_ade20k.pth --export onnx/upernet_internimage_h_896_160k_ade20.onnx
      ```
      
      - 参数说明
        
        - --config：配置文件路径
        
        - --ckpt：权重文件路径
        
        - --export：导出的ONNX模型路径
   
   3. 请访问[msit推理工具](https://gitcode.com/ascend/msit/tree/master/msit/)代码仓，根据README文档进行工具安装benchmark和surgeon
   
   4. 使用ATC工具将ONNX模型转为OM模型
      
      1. 配置环境变量
         
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
      
      2. 执行命令查看芯片名称($(chip_name))
         
         > ```
         > npu-smi info
         > #该设备芯片名为Ascend310P3 （自行替换）
         > 回显如下：
         > +-------------------+-----------------+------------------------------------------------------+
         > | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         > | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         > +===================+=================+======================================================+
         > | 0       310P3     | OK              | 15.8         42                0    / 0              |
         > | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         > +===================+=================+======================================================+
         > | 1       310P3     | OK              | 15.4         43                0    / 0              |
         > | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         > +===================+=================+======================================================+
         > ```
      
      3. 执行ATC命令将ONNX模型转为OM模型
         
         ```
         atc --model=onnx/upernet_internimage_h_896_160k_ade20.onnx --framework=5 --output=om/upernet_internimage_h_896_160k_ade20 --input_format=NCHW --input_shape="input:-1,3,-1,-1" --soc_version=Ascend${chip_name} --keep_dtype exceptionlist.cfg
         ```
         
         - 参数说明
           
           - --model：为ONNX模型文件路径
           
           - --framework：5代表ONNX模型
           
           - --output：输出的OM模型路径
           
           - --input_format：输入数据的格式
           
           - --input_shape：输入数据的shape
           
           - --log：日志输出级别设置
           
           - --keep_dtype: 指定部分算子使用FP32运行
           
           - --soc_version：指定目标芯片型号

2. 开始推理验证
   
   1. 数据预处理
      
      执行如下命令以开始数据预处理
      
      ```
      python3 preprocess_data.py --config configs/ade20k/upernet_internimage_h_896_160k_ade20k.py --output ./data_after_preprocess/
      ```
      
      - 参数说明
        
        - --config：配置文件路径
        
        - --output：数据经预处理后的输出路径
   
   2. 执行离线推理
      
      ```
      python3 -m ais_bench --model om/upernet_internimage_h_896_160k_ade20_linux_aarch64.om --input ./data_after_preprocess --output ./ --output_dirname om_output --outfmt NPY --auto_set_dymshape_mode 1 --outputSize "100000000"
      ```
      
      - 参数说明：
        
        - ·--model：离线推理所使用的OM模型路径
        
        - --input：离线推理所使用的数据集
        
        - --output：推理结果保存目录
        
        - --output_dirname：推理结果保存子目录
        
        - --outfmt：输出数据的格式。取值可以是："NPY","BIN","TXT"。本项目暂时只支持NPY格式输出
        
        - --auto_set_dymshape_mode：自动设置动态shape模型。1或true为开启，0或false为关闭
        
        - --outputSize：指定模型的输出数据所占内存大小
   
   3. 数据后处理
      
      执行如下命令以开始数据后处理并获得OM模型的精度。
      
      ```
      python3 post_process.py --config configs/ade20k/upernet_internimage_h_896_160k_ade20k.py --input ./om_output
      ```
      
      - 参数说明：
        
        - --config：配置文件路径
        
        - --input：后处理脚本的输入数据的路径
      
      上面的脚本运行完毕将会打印精度数据
   
   4. 性能推理
      
      运行以下命令获取OM模型的性能数据
      
      ```
      python3 -m ais_bench --model om/upernet_internimage_h_896_160k_ade20_linux_aarch64.om --outputSize "100000000" --loop 100 --dymShape "input:1,3,896,896"
      ```

# 模型推理性能 & 精度<a id="performance"></a>

后处理完成后会打印精度数据，精度参考下列数据。

| 芯片型号     | 模型                                    | mIoU  | aAcc  | mAcc | 性能        |
| -------- | ------------------------------------- | ----- | ----- | ---- | --------- |
| 300I PRO | upernet_internimage_h_896_160k_ade20k | 59.54 | 86.55 | 71.8 | 2067.32ms |
