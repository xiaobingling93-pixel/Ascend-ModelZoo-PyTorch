# GLIP 静态-推理指导


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
  | image    | RGB_FP32 | 1 x 3 x 784 x 1344 | NCHW         |
  | text_id     | INT64 | 1 x 256 | ND        |
  | image    | FLOAT32 | 1 x 256 | ND         |
  





# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 8.0.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.13.1   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/microsoft/GLIP.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

3. 打补丁。

   ```
   patch -p2 < ./glip.patch
   ```


4. 转移文件夹位置。

    ```
    mv onnx_model.py ./GLIP/maskrcnn_benchmark/modeling/detector
    mv pth2onnx.py ./GLIP/
    mv inference.py ./GLIP/
    mv fix_onnx.py ./GLIP/
    cd GLIP
    ```

## 准备数据集<a name="section183221994411"></a>

1. 该模型静态形态不包含数据集推理，精度对比输出余弦相似度


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

         python pth2onnx.py --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight=glip_tiny_model_o365_goldg.pth --model_type='lang'

         python pth2onnx.py --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight=glip_tiny_model_o365_goldg.pth --model_type='rpn_head' 
         ```

         分别获得glip_backbone.onnx文件, glip_language.onnx文件和glip_rpn.onnx文件。

      2. 使用convert.py和fix_onnx.py脚本对onnx模型进行优化(按照链接安装auto-optimizer[[auto-optimizer链接](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)])
         ```
         python3 fix_onnx.py glip_rpn.onnx glip_rpn_new.onnx
         python3 fix_onnx.py glip_language.onnx glip_language_new.onnx
         ```
         获得glip_language_new.onnx文件和glip_rpn_new.onnx文件

      3. 使用onnx-simplifier对模型简化(按照链接安装onnx-simplifier[[onnx-simplifier链接](https://github.com/daquexian/onnx-simplifier)])
         ```
         python3 -m onnxsim glip_backbone.onnx glip_backbone_sim.onnx
         ```
         获得简化后的glip_backbone_sim.onnx模型。


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
         atc --model glip_backbone_sim.onnx --output glip_backbone --framework=5 --soc_version=Ascend310P3 --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance 
   
         atc --model glip_language_new.onnx --output glip_language --framework=5 --soc_version=Ascend310P3
   
         atc --model glip_rpn_new.onnx --output glip_rpn --framework=5 --soc_version=Ascend310P3
   
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape\_range：设置动态输入数据的shape。
           -   --soc\_version：处理器型号。
           -   --fusion_switch_file: 对于融合规则开关。

           运行成功后生成om模型文件。

2. 开始推理验证。

   1. 安装msit推理工具。

      请访问[msit推理工具](https://gitee.com/ascend/msit.git)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 inference.py --config_file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight=glip_tiny_model_o365_goldg.pth  
        ```

        -   参数说明：

             -   config-file：模型参数。
             -   weight：模型权重。

        推理后打屏15个输出的CPU推理结果和NPU推理结果的余弦相似度。

   3. 性能推理。
      ```
      python3 -m ais_bench --om-model glip_language.om --loop 100
      python3 -m ais_bench --om-model glip_backbone.om --loop 100
      python3 -m ais_bench --om-model glip_rpn.om --loop 100
      ```
      获得三段模型分别的性能数据

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 模型 | 性能 |
| --------- | ---------------- | ---------- | ---------- |
|     Ascend310P3      |        1          |    glip_backbone.om       |       39.38 ms          |
|     Ascend310P3      |        1          |      glip_language.om     |        3.47 ms        |
|     Ascend310P3      |        1          |     glip_rpn.om      |      184.57   ms        |