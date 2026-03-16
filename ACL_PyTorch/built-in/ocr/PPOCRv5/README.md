# PPOCRv5(ONNX)-推理指导

- [PPOCRv5(ONNX)-推理指导](#PPOCRv5(ONNX)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [获取权重](#获取权重)
  - [执行推理](#模型推理)
  - [精度测试](#精度测试)
  - [结果展示](#结果展示)

******

# 概述
PP-OCRv5 是PP-OCR新一代文字识别解决方案，该方案聚焦于多场景、多文字类型的文字识别。在文字类型方面，PP-OCRv5支持简体中文、中文拼音、繁体中文、英文、日文5大主流文字类型，在场景方面，PP-OCRv5升级了中英复杂手写体、竖排文本、生僻字等多种挑战性场景的识别能力。PPOCRv5通过OCR产线解决文字识别任务，通过调用文本检测模块以及文本识别模块，提取图片中的文字信息以文本形式输出。

本文档介绍了PPOCRv5模型的部署流程，包括推理环境准备、模型部署、功能验证，旨在帮助用户快速完成模型部署和验证。

- 版本说明：
  
  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  commit_id=ac86ace5a65c41441cba6cddfee1820c53ad1664
  model_name=PPOCRv5
  ```

# 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1**  版本配套表
  
  | 配套                                                      | 版本          | 环境准备指导                                                                                        |
  | ------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                   | 25.2.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                    | 8.5.0       | -                                                                                             |
  | Python                                                  | 3.11        | -                                                                                             |
  | 说明：Atlas 300I Pro 推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |

# 快速上手

## 获取源码

1. 安装依赖  
   
   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/ocr/PPOCRv5
   python -m pip install paddlepaddle==3.2.2
   git clone --depth 1 --branch v3.3.2 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR
   pip3 install -e .
   git apply ../paddleocr.patch
   cd ..
   pip install -r requirements.txt
   ```

2. 安装msit工具
   ```
   pip install msit
   msit install surgeon
   wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl
   wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl
   pip install ais_bench-0.0.2-py3-none-any.whl
   pip install aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl
   ```

3. 配置CANN环境变量

   执行以下命令激活CANN环境变量。注意：该命令中文件路径仅供参考，请以实际安装环境配置环境变量。详细介绍请参见[《CANN 开发辅助工具指南 (推理)》](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=openEuler)。
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

## 获取权重

PPOCRv5产线主要包含文本检测模块（det）以及文本识别模块（rec），运行以下指令，分别下载det模型[Det Model weights](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar)以及rec模型[Rec Model weights](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_server_rec_infer.tar)权重文件，并进行解压
   ```
   wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar
   wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_server_rec_infer.tar
   tar -xf PP-OCRv5_server_det_infer.tar
   tar -xf PP-OCRv5_server_rec_infer.tar
   ```

## 模型推理
### 模型转换
1. 导出onnx模型
   ```
   paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_det_infer --onnx_model_dir ./PP-OCRv5_server_det_infer
   paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_rec_infer --onnx_model_dir ./PP-OCRv5_server_rec_infer
   ```

2. onnx模型优化
   通过onnxsim以及auto_optimizer对onnx模型进行优化
   ```
   onnxsim PP-OCRv5_server_det_infer/inference.onnx PP-OCRv5_server_det_infer/inference_sim.onnx
   onnxsim PP-OCRv5_server_rec_infer/inference.onnx PP-OCRv5_server_rec_infer/inference_sim.onnx
   python onnx_optimizer.py PP-OCRv5_server_det_infer/inference_sim.onnx PP-OCRv5_server_det_infer/inference_opt.onnx
   python onnx_optimizer.py PP-OCRv5_server_rec_infer/inference_sim.onnx PP-OCRv5_server_rec_infer/inference_opt.onnx
   ```

3. 导出om模型
   执行`npu-smi info`查看芯片名称，并赋值为${soc_version}，执行ATC命令，生成det和rec的om模型，结果存放于各自权重路径，文件名均为inference_{arch}.om，模型{arch}后缀为当前使用的CPU操作系统。
   ```
   atc --model=./PP-OCRv5_server_det_infer/inference_opt.onnx --framework=5 --output=./PP-OCRv5_server_det_infer/inference --soc_version=Ascend${soc_version} --input_shape "x:-1,3,-1,-1"
   atc --model=./PP-OCRv5_server_rec_infer/inference_opt.onnx --framework=5 --output=./PP-OCRv5_server_rec_infer/inference --soc_version=Ascend${soc_version} --input_shape "x:-1,3,48,-1"
   ```
### 运行推理脚本
获取示例图像存放在工作路径，并执行推理脚本，脚本通过调用PaddleOCR接口进行产线推理，通过指定文本检测模块以及文本识别模块，解决了文本识别任务，将图片中的文字信息以文本形式输出。
   ```
   wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
   python3 infer.py --image_dir general_ocr_002.png
   ```

- 参数说明
  - image_dir: 待预测数据，可以是图像文件或者包含图片的本地目录
  - det_model_name: 文本检测模型的名称，默认为PP-OCRv5_server_det
  - det_model_dir: 文本检测模型的目录路径，默认为PP-OCRv5_server_det_infer
  - rec_model_name: 文本识别模型的名称，默认为PP-OCRv5_server_rec
  - rec_model_dir: 文本识别模型的目录路径，默认为PP-OCRv5_server_rec_infer

推理执行完成后，解析结果存放于`output`目录，目录包含存有各项中间结果的json文件以及可视化结果图像。


## 精度测试
参照paddleocr的精度测试流程[det](https://www.paddleocr.ai/main/version3.x/module_usage/text_detection.html)、[rec](https://www.paddleocr.ai/main/version3.x/module_usage/text_recognition.html)，分别对文本检测以及识别进行精度测试，首先进入PaddleOCR目录
   ```
   cd PaddleOCR
   ```
### 文本检测模型精度验证
1. 准备数据集
   ```
   wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar
   tar -xf ocr_det_dataset_examples.tar
   ```
2. 模型评估
   使用如下命令得到精度指标：
   ```
   python3 tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_server_det.yml \
      -o Global.pretrained_model=../PP-OCRv5_server_det_infer/inference_linux_aarch64.om \
      Eval.dataset.data_dir=./ocr_det_dataset_examples \
      Eval.dataset.label_file_list='[./ocr_det_dataset_examples/val.txt]' \
      Global.use_gpu=False
   ```
   预期输出precision、recall以及hmean指标，这里选用hmean作为文本检测模型的精度指标，其表示precision与recall的调和平均，能够更好地描述检测模型的整体质量。在ocr_det_dataset_examples数据集上，PP-OCRv5_server_det模型预期输出的hmean指标为0.5287。

### 文本识别模型精度验证
1. 准备数据集
   ```
   wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
   tar -xf ocr_rec_dataset_examples.tar
   ```

2. 模型评估
   使用如下命令得到精度指标：
   ```
   python3 tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
      -o Global.pretrained_model=../PP-OCRv5_server_rec_infer/inference_linux_aarch64.om \
      Eval.dataset.data_dir=./ocr_rec_dataset_examples \
      Eval.dataset.label_file_list='[./ocr_rec_dataset_examples/val.txt]' \
      Global.use_gpu=False
   ```
   预期输出acc以及norm_edit_dis两个指标，分别表示准确率以及归一化编辑距离，这里选用acc作为文本识别模型的精度指标，PP-OCRv5_server_rec模型在ocr_rec_dataset_examples数据集上的预期输出为0.7227。

### 性能测试
对于文本检测模型，输入的数据要求长宽为32的整数倍，我们以（1,3,960,960）的输入shape为例，运行如下命令测试推理性能，
   ```
   python -m ais_bench --model ../PP-OCRv5_server_det_infer/inference_linux_aarch64.om --outputSize "1000000000" --dymShape "x:1,3,960,960" --loop 100
   ```
对于文本识别模型，以（1,3,48,320）shape为例，运行如下命令测试性能
   ```
   python -m ais_bench --model ../PP-OCRv5_server_rec_infer/inference_linux_aarch64.om --outputSize "1000000000" --dymShape "x:1,3,48,320" --loop 100
   ```

## 结果展示
   文本检测模型和文本识别模型在示例数据集上的的精度和性能数据展示在下表，其中文本检测模型的精度指标为hmeans，文本识别模型的精度指标为acc：
   
   EP模式: Atlas 300I DUO;
   RC模式: Ascend 310P RC 176T

   |模型|hmeans/acc|hmeans/acc（A10）|性能（ms）(EP模式)|性能（ms）(RC模式)|
   |------|------|------|------|------|
   |PP-OCRv5_server_det|52.87%|52.87%|35.01 |26.85 |
   |PP-OCRv5_server_rec|72.27%|72.27%|2.63 |6.09 |
