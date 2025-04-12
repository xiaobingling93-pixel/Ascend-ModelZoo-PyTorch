# InternImage Detection 推理指导

- [概述](#summary)
  
  - [输入数据](#input_data)

- [推理环境准备](#env_setup)

- [快速上手](#quick_start)
  
  - [获取源码](#get_code)
  
  - [下载数据集](#download_data)
  
  - [模型推理](#infer)

- [模型推理性能 & 精度](#performance)

# 概述<a id="summary"></a>

InternImage 是一个由上海人工智能实验室、清华大学等机构的研究人员提出的基于卷积神经网络（CNN）的视觉基础模型。与基于 Transformer 的网络不同，InternImage 以可变形卷积 DCNv3 作为核心算子，使模型不仅具有检测和分割等下游任务所需的动态有效感受野，而且能够进行自适应的空间聚合。此指导仅针对InternImage项目下的以InternImage-XL为backbone，method使用Cascade，schd为3×的模型。该模型使用box mAP与mask mAP作为评价指标。

- 版本说明:
  
  ```
  url=https://github.com/OpenGVLab/InternImage
  commit_id=41b18fd85f20a4f85c0a1e6b1d5f97303aab1800
  model_name=InternImage
  ```

## 输入数据<a id="input_data"></a>

InternImage使用公共数据集COCO进行推理

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
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/InternImage_detection_for_Pytorch
   ```

2. 获取模型仓**InternImage**源码和依赖仓**mmdet**源码
   
   ```
   git clone https://github.com/open-mmlab/mmdet.git
   git clone https://github.com/OpenGVLab/InternImage.git
   cd mmdetection
   git reset --hard cfd5d3a985b0249de009b67d04f37263e11cdf3d
   cd ../InternImage
   git reset --hard 41b18fd85f20a4f85c0a1e6b1d5f97303aab1800
   cd ..
   ```

3. 转移文件夹位置
   
   ```
   mv internimage_det.patch InternImage/detection/
   mv *.py InternImage/detection/
   mv exceptionlist.cfg InternImage/detection/
   mv mmdet.patch mmdetection/mmdet/
   ```

4. 安装依赖
   
   ```
   pip3 install -r requirement.txt
   ```

5. 更换当前路径并打补丁，修改完mmseg源码后进行安装
   
   ```
   cd mmdetection/mmdet/
   patch -p2 < mmdet.patch
   cd ..
   pip3 install -v -e .
   
   cd ../../InternImage/detection/
   patch -p2 < internimage_det.patch
   ```

## 下载数据集<a id="download_data"></a>

    使用下面的链接下载数据集并解压放在InternImage/detection/data目录下

> [COCO数据集下载](https://cocodataset.org/#download)

    确保data下的路径结构如下

```
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── val2017
```

## 模型推理<a id="infer"></a>

1. 使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件
   
   1. 下载权重文件并放到InternImage/detection/ckpt下
      
      > [ckpt文件下载](https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth)
   
   2. 数据预处理
      
      执行如下命令以开始数据预处理。预处理脚本会另外生成一个img_shape.npy文件，将会用于离线推理。由于此项目源码中存在不同尺寸的图片输出的多尺度特征走不同的处理的情况，而ONNX只能记录其中一种处理路径。为了所有尺寸的图片能有一致的处理方法，此项目在预处理时将所有图片统一缩放为1216*1216尺寸
      
      ```
      python3 preprocess.py --config configs/coco/cascade_internimage_xl_fpn_3x_coco.py --data_output data_after_preprocess --force_img_shape 1216,1216
      ```
      
      - 参数说明
        
        - --config：配置文件路径
        
        - --data_output：数据经预处理后的输出路径
        
        - --force_img_shape：原图经强制缩放后的尺寸
   
   3. 导出onnx文件
      
      确认当前路径为InternImage/detection并执行如下命令导出onnx文件
      
      ```
      python export2onnx.py --config configs/coco/cascade_internimage_xl_fpn_3x_coco.py --ckpt ckpt/cascade_internimage_xl_fpn_3x_coco.pth --export onnx/cascade_internimage_xl_fpn_3x_coco.onnx --data ./data_after_preprocess --img_shape_path ./img_shape
      ```
      
      - 参数说明
        
        - --config：配置文件路径
        
        - --ckpt：权重文件路径
        
        - --export：导出的ONNX模型路径
        
        - --data：前处理后的图像数据的路径
        
        - --img_shape_path：存储前处理后的图像shape
   
   4. 请访问[msit推理工具](https://gitee.com/ascend/msit/tree/master/msit/)代码仓，根据README文档进行工具安装benchmark和surgeon
   
   5. 使用ATC工具将ONNX模型转为OM模型
      
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
         atc --model=onnx/cascade_internimage_xl_fpn_3x_coco.onnx --framework=5 --output=om/cascade_internimage_xl_fpn_3x_coco --input_format=NCHW --input_shape="data:1,3,1216,1216" --soc_version=Ascend${chip_name} --keep_dtype exceptionlist.cfg
         ```
         
         - 参数说明
           
           - --model：为ONNX模型文件路径
           
           - --framework：5代表ONNX模型
           
           - --output：输出的OM模型路径
           
           - --input_format：输入数据的格式
           
           - --input_shape：输入数据的shape
           
           - --keep_dtype: 指定部分算子使用FP32运行
           
           - --soc_version：指定目标芯片型号

2. 开始推理验证
   
   1. 执行离线推理
      
      ```
      python3 -m ais_bench --model om/cascade_internimage_xl_fpn_3x_coco.om --input ./data_after_preprocess,./img_shape --output ./ --output_dirname om_output --outfmt NPY
      ```
      
      - 参数说明：
        
        - --model：离线推理所使用的OM模型路径
        
        - --input：离线推理所使用的数据集
        
        - --output：推理结果保存目录
        
        - --output_dirname：推理结果保存子目录
        
        - --outfmt：输出数据的格式。取值可以是："NPY","BIN","TXT"。本项目暂时只支持NPY格式输出
   
   2. 数据后处理
      
      执行如下命令以开始数据后处理并获得OM模型的精度
      
      ```
      python3 postprocess.py --config configs/coco/cascade_internimage_xl_fpn_3x_coco.py --ckpt ckpt/cascade_internimage_xl_fpn_3x_coco.pth --om_output om_output --eval bbox segm --batch_size 100 --force_img_shape 1216,1216
      ```
      
      - 参数说明：
        
        - --config：配置文件路径
        
        - --ckpt：权重文件的路径
        
        - --om_output：OM模型的输出路径，也即后处理脚本的输入数据路径
        
        - --eval：精度评估指标
        
        - --batch_size：后处理的时候每次加载的图片数量，取决于内存大小。每张图片大约需要120M的内存空间。默认为100
        
        - --force_img_shape：预处理时原图经强制缩放后的尺寸
      
      上面的脚本运行完毕将会打印精度数据
   
   3. 性能推理
      
      运行以下命令获取OM模型的性能数据
      
      ```
      python3 -m ais_bench --model om/cascade_internimage_xl_fpn_3x_coco.om --loop 100
      ```
      
      * 参数说明：
        
        * --model：OM模型路径
        
        * --loop：离线推理循环次数

# 模型推理性能 & 精度<a id="performance"></a>

后处理完成后会打印精度数据，精度参考下列数据。

| 芯片型号     | 模型                                 | box mAP | seg mAP | 性能        |
| -------- | ---------------------------------- | ------- | ------- | --------- |
| 300I PRO | cascade_internimage_xl_fpn_3x_coco | 0.556   | 0.486   | 1614.70ms |
