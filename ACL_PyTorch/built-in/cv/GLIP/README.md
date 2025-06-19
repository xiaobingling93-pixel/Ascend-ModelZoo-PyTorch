# GLIP-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

标准的 object detection 模型只能推理固定的对象类别，如COCO，而这种人工标注的数据扩展成本很高。GLIP将 object detection 定义为 phrase grounding，可以推广到任何目标检测任务。

CLIP和ALIGN在大规模图像-文本对上进行跨模态对比学习，可以直接进行开放类别的图像分类。GLIP继承了这一研究领域的语义丰富和语言感知的特性，实现了SoTA对象检测性能，并显著提高了对下游检测任务的可迁移能力。


- 参考实现：

  ```
  url= https://github.com/microsoft/GLIP
  commit_id=a5f302bfd4c5c67010e29f779e3b0bde94e89985
  model_name= GLIP
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x height x weight | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | bbox  | FLOAT32  | N x 4 | ND           |
  | score  | FLOAT32  | N x 1 | ND           |
  | label  | FLOAT32  | N x 1 | ND           |    


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 7.0.T800 | -                                                            |
  | Python                                                       | 3.9   | -                                                            |
  | PyTorch                                                      | 1.13.1   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码
   
   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/GLIP
   ```

1. 获取源码。

   ```
   git clone https://github.com/microsoft/GLIP.git
   cd GLIP
   git reset --hard a5f302bfd4c5c67010e29f779e3b0bde94e89985
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirement.txt
   ```

3. 打补丁。

   ```
   patch -p1 < ./glip.patch
   ```


4. 转移文件夹位置。

    ```
    mkdir -p backbone/model
    mkdir -p rpn_head/model
    mkdir -p select/model
    mv onnx_model.py ./GLIP/maskrcnn_benchmark/modeling/detector
    mv pth2onnx.py ./GLIP/
    mv inference.py ./GLIP/
    mv backbone ./GLIP
    mv rpn_head ./GLIP
    mv select ./GLIP
    cd GLIP
    ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   本模型支持coco2017验证集。用户需自行获取[数据集](http://images.cocodataset.org/zips/val2017.zip)与[标注文件](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)，将annotations_trainval2017.zip和val2017.zip上传并解压到如下目录结构：

   ```
   GLIP
   └── DATASET 
        └── coco
              ├── annotations    
              └── val2017
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取GLIP模型权重
       https://huggingface.co/GLIPModel/GLIP/blob/main/glip_tiny_model_o365_goldg.pth
       将权重放在GLIP目录下


       获取bert-base-uncase权重

       https://huggingface.co/bert-base-uncased/tree/main
       将bert-base-uncase文件夹放至GLIP目录下

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         python pth2onnx.py --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight=glip_tiny_model_o365_goldg.pth --model_type='backbone'

         python pth2onnx.py --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight=glip_tiny_model_o365_goldg.pth --model_type='select'

         python pth2onnx.py --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight=glip_tiny_model_o365_goldg.pth --model_type='rpn_head' TEST.IMS_PER_BATCH 1
         ```

         分别获得./backbone/model/glip_backbone.onnx文件, ./select/model/glip_select.onnx文件和./rpn_head/model/glip_rpn_head.onnx文件。

      2. 使用convert.py脚本对glip_rpn_head.onnx进行优化
         ```
         python3 ./rpn_head/model/convert.py
         ```
         获得glip_rpn_head_new.onnx文件


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh  #默认位置；须按脚本实际位置使用
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
         atc --model="./backbone/model/glip_backbone.onnx" --output=./backbone/model/glip_backbone --framework=5 --soc_version=Ascend${chip_name} --input_shape_range="images:[1,3,300~1400,500~1400]"

         atc --model="./select/model/glip_select.onnx" --output=./select/model/glip_select --framework=5 --soc_version=Ascend${chip_name} --input_shape_range="input_1:[1,256,40~200,60~200];input_2:[1,256,20~100,30~100];input_3:[1,256,10~50,15~50];input_4:[1,256,5~25,7~25];input_5:[1,256,3~15,4~15]"

         atc --model="./rpn_head/model/glip_rpn_head_new.onnx" --output=./rpn_head/model/glip_rpn_head --framework=5 --soc_version=Ascend${chip_name} --input_shape_range="feature_1:[1,256,40~200,60~200];feature_2:[1,256,20~100,30~100];feature_3:[1,256,10~50,15~50];feature_4:[1,256,5~25,7~25];feature_5:[1,256,3~15,4~15]" --fusion_switch_file=./rpn_head/model/fusion_switch.cfg
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape\_range：设置动态输入数据的shape。
           -   --soc\_version：处理器型号。
           -   --fusion_switch_file: 对于融合规则开关。

           运行成功后生成./backbone/glip_backbone_linux_aarch64.om, ./select/model/glip_select_linux_aarch64.om和./rpn_head/model/glip_rpn_head_linux_aarch64.om模型文件。

2. 开始推理验证。

   1. 安装msit推理工具。

      请访问[msit推理工具](https://gitee.com/ascend/msit/tree/master/msit/)代码仓，根据readme文档安装benchmark组件。安装完后可以使用`msit check all`看安装的组件下是否显示OK来判断是否成功安装。

   2. 执行推理。

        ```
        python3 inference.py --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight=glip_tiny_model_o365_goldg.pth --output_folder=./result TEST.IMS_PER_BATCH 1 MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection  
        ```

        -   参数说明：

             -   config-file：模型参数。
             -   weight：模型权重。
             -   output_folder：结果保存路径。
               	...

        推理后的精度打屏，保存在output_folder下。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|     300I Pro      |        1          |     coco       |     46.3%       |                 |