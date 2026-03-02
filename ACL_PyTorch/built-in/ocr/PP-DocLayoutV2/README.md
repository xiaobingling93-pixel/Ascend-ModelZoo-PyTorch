# PP-DocLayoutV2(ONNX)-推理指导

- [PP-DocLayoutV2(ONNX)-推理指导](#PP-DocLayoutV2(ONNX)-推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [环境配置](#环境配置)
  - [获取权重](#获取权重)
  - [执行推理](#模型推理)
  - [精度测试](#精度测试)
  - [结果展示](#结果展示)

******

# 概述
版面分析是文档解析系统中的重要组成环节，目标是对文档的整体布局进行解析，准确检测出其中所包含的各种元素（例如，文本段落、标题、图片、表格、公式等），并恢复这些元素正确的阅读顺序。PP-DocLayoutV2是一个专门用于布局分析的轻量级模型，专注于元素检测、分类和读取顺序预测，属于PaddleOCRVL模型中的一个核心组件。

本文档介绍了PP-DocLayoutV2模型的部署流程，包括推理环境准备、模型部署、功能验证，旨在帮助用户快速完成模型部署和验证。

- 版本说明：
  
  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  commit_id=95dc316a8490a350c1d1fcf3bc80bf83d4e52450
  model_name=PP-DocLayoutV2
  ```

# 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1**  版本配套表
  
  | 配套                                                    | 版本         | 环境准备指导                                                                                   |
  | ------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                               | 25.2.RC1    | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                    | 8.3.0       | -                                                                                             |
  | Python                                                  | 3.11        | -                                                                                             |
  | 说明：Atlas 300I Pro、Atlas 800I A2推理卡请以CANN版本选择实际固件与驱动版本。 | \           | \                                                                                             |
注意：在部署PaddleOCR-VL模型时，建议将PP-DocLayoutV2与PaddleOCR-VL-0.9B模型的**环境隔离**，以防依赖冲突

# 快速上手

## 环境配置

1. 安装依赖  
   
   ```
   git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/ocr/PP-DocLayoutV2
   pip install -r requirements.txt
   workdir=$(pwd)
   source_path=$(pip show paddlex | grep Location | awk '{print $2}')
   cd ${source_path}/paddlex
   patch -p1 < ${workdir}/paddlex.patch
   cd ${workdir}
   ```
   可能缺少opencv组件
   ```
   apt-get update
   apt-get install -y libgl1 libglib2.0-0
   ```
2. 安装msit工具以及相关组件
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

下载PP-DocLayoutV2[模型权重](https://www.modelscope.cn/models/PaddlePaddle/PP-DocLayoutV2/files)，并放置于本地目录`PP-DocLayoutV2`
   ```
   mkdir PP-DocLayoutV2
   modelscope download --model PaddlePaddle/PP-DocLayoutV2 --local_dir ./PP-DocLayoutV2
   ```

## 模型推理
### 模型转换
1. 导出onnx模型
   ```
   paddlex --paddle2onnx --paddle_model_dir PP-DocLayoutV2/ --onnx_model_dir PP-DocLayoutV2
   ```

2. onnx模型优化
   通过auto_optimizer对onnx模型进行改图优化。
   ```
   python3 -m auto_optimizer optimize PP-DocLayoutV2/inference.onnx PP-DocLayoutV2/inference_opt.onnx
   ```

3. 导出om模型

   参照[ATC工具指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/atctool/atlasatc_16_0005.html)，执行`npu-smi info`查看芯片名称，并赋值为${soc_version}，执行ATC命令，生成PP-DocLayoutV2的om模型，文件名为inference_{arch}.om，模型{arch}后缀为当前使用的CPU操作系统。
   ```
   atc --model=PP-DocLayoutV2/inference_opt.onnx --framework=5 --output=PP-DocLayoutV2/inference --soc_version=Ascend${soc_version} --input_shape "im_shape:-1,2;image:-1,3,800,800;scale_factor:-1,2"
   ```
   注意：atc工具失效可能是python动态库配置问题，参考如下指令添加动态库路径，具体设置以实际python路径为准
   ```
   export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:$LD_LIBRARY_PATH
   ```

### 运行推理脚本infer.py
获取示例图像存放在工作路径，并执行推理脚本，脚本通过调用PaddleOCR接口进行产线推理，通过指定文本检测模块以及文本识别模块，解决了文本识别任务，将图片中的文字信息以文本形式输出。
   ```
   wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg
   python3 infer.py --data_dir layout.jpg
   ```

- 参数说明
  - data_dir: 待预测数据，可以是图像文件或者包含图片的本地目录
  - model_name: 布局检测模型的名称，默认为PP-DocLayoutV2
  - model_dir: 布局检测模型的目录路径，默认为PP-DocLayoutV2

推理执行完成后，解析结果存放于`output`目录，目录包含存有各项中间结果的json文件以及可视化结果图像。


## 精度测试
PP-DocLayoutV2是PaddleOCR-VL模型的核心组成部分，为了验证其有效性，结合PaddleOCR-VL-0.9B模型进行OCR识别任务检验精度。vllm的部署流程参照[PaddleOCR-VL](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/PaddleOCR-VL.html)，部署完成PaddleOCR-VL-0.9B后，启动服务化，对外接口为8000。

要注意，建议将PP-DocLayoutV2环境与vllm环境隔离，以防遇到依赖冲突。

### 获取数据集

创建数据集目录`OmniDocBenchV1.5`，下载多样性文档解析评测集`OmniDocBench`数据集的[pdfs和标注](https://opendatalab.com/OpenDataLab/OmniDocBench)，解压并放置在`OmniDocBenchV1.5`目录下
文件目录格式大致如下：
   ```
   📁 workdir
    ├── infer.py
    ├── ……
    └── 📁 OmniDocBenchV1.5
        ├── OmniDocBench.json
        └── 📁 pdfs
            └── ***.pdf
   ```
### 推理结果 
运行测试脚本test.py

```
python3 test.py --data_dir=OmniDocBenchV1.5/pdfs --output_path=OmniDocBenchV1.5_out_pdf --layout_detection_model_name=PP-DocLayoutV2 --layout_detection_model_dir=PP-DocLayoutV2 --vllm_ip="http://127.0.0.1:8000/v1"
```
- 参数说明
  - data_path: 数据集路径
  - output_path: markdown文件存放路径
  - layout_detection_model_name: 布局检测模型名称，默认为PP-DocLayoutV2
  - layout_detection_model_dir: 布局检测模型路径，默认为PP-DocLayoutV2
  - vllm_ip: PaddleOCR-VL-0.9B模型对外接口，默认为"http://127.0.0.1:8000/v1"

推理执行完成后，解析结果存放于`OmniDocBenchV1.5_out_pdf`目录。按照如下步骤汇总解析结果，使得文件名与数据集标注对应
```
mkdir OmniDocBenchV1.5_out_pdf/end2end
cp OmniDocBenchV1.5_out_pdf*.md OmniDocBenchV1.5_out_pdf/end2end
for f in OmniDocBenchV1.5_out_pdf/end2end/*_0.md; do mv "$f" "${f%_0.md}.md"; done
```

### 测评环境构建
1. 获取测评源码并构建环境

   - 安装OmniDocBench基础环境
   
   ```
   git clone https://github.com/opendatalab/OmniDocBench.git
   cd OmniDocBench
   git reset --hard 523fd1d529c3e9d0088c662e983aa70fb9585c9a
   conda create -n omnidocbench python=3.10
   conda activate omnidocbench
   pip install -r requirements.txt
   ```

   - 公式精度指标CDM需要额外安装环境

   step.1 install nodejs
   ```
   wget https://nodejs.org/dist/v16.13.1/node-v16.13.1-linux-arm64.tar.xz
   tar -xf node-v16.13.1-linux-arm64.tar.xz
   mv node-v16.13.1-linux-arm64/* /usr/local/nodejs/
   ln -s /usr/local/nodejs/bin/node /usr/local/bin
   ln -s /usr/local/nodejs/bin/npm /usr/local/bin
   node -v
   ```

   step.2 install imagemagic
   ```
   git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.2
   cd ImageMagick-7.1.2
   apt-get update && apt-get install -y libpng-dev zlib1g-dev
   apt-get install -y ghostscript
   ./configure
   make
   sudo make install
   sudo ldconfig /usr/local/lib
   convert --version
   ```

   step.3 install latexpdf
   ```
   sudo apt-get install texlive-full
   ```

   step.4 install python requriements
   ```
   pip install -r metrics/cdm/requirements.txt
   ```

2. 测评配置修改

   修改`OmniDocBench`测评代码中的config文件，具体来说，我们使用端到端测评配置，修改configs/end2end.yaml文件中的ground_truth的data_path为下载的OmniDocBench.json路径，修改prediction的data_path中提供整理的推理结果的文件夹路径，如下：
   ```
   # -----以下是需要修改的部分 -----
    display_formula:
      metric:
        - Edit_dist
        - CDM       ### 安装好CDM环境后，可以在config文件中设置并直接计算
        - CDM_plain
   ...
   dataset:
      dataset_name: end2end_dataset
      ground_truth:
      data_path: ../OmniDocBenchV1.5/OmniDocBench.json
      prediction:
      data_path: ../OmniDocBenchV1.5_out_pdf/end2end
   ```

3. 精度测量结果

   配置好config文件后，只需要将config文件作为参数传入，运行以下代码即可进行评测：
   ```
   python pdf_validation.py --config ./configs/end2end.yaml
   ```
   评测结果将会存储在result目录下，Overall指标的计算方式为:
   $$\text{Overall} = \frac{(1-\textit{Text Edit Distance}) \times 100 + \textit{Table TEDS} +\textit{Formula CDM}}{3}$$

   运行overall_metric.py可以得到精度结果：
   ```
   cd ..
   python overall_metric.py
   ```

## 结果展示
   在OmniDocBenchV1.5数据集上的精度和性能数据分别为：

   |模型|芯片|overall|官方精度|性能(s)|
   |------|------|------|------|------|
   |PaddleOCR-VL|300I DUO|92.43%|92.86%|12.10|
