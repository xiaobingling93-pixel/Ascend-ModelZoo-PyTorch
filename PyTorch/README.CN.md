# 欢迎使用ModelZoo-PyTorch

为方便更多开发者使用ModelZoo-PyTorch，我们将持续增加典型网络和相关预训练模型。下面目录中我们罗列出一些重点模型。如果您有任何需求，请在[modelzoo/issues](https://gitee.com/ascend/modelzoo/issues)提交issue，我们会及时处理。

## 如何贡献

在开始贡献之前，请先阅读[CONTRIBUTING](https://gitee.com/ascend/modelzoo/blob/master/CONTRIBUTING.md)。
谢谢！

## 安装依赖

使用ModelZoo-PyTorch之前，请参考[软件版本配套表](#软件版本配套表)，安装最新昇腾软件栈。

<table border="0">
  <tr>
    <th>依赖软件</th>
    <th>软件安装指南</th>
  </tr>

  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">《 <a href="https://www.hiascend.com/document/detail/zh/Pytorch/600/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>

## 软件版本配套表
💡 使用ModelZoo-PyTorch中的PyTorch训练模型前，请先确认PyTorch和CANN版本，然后匹配对应的PyTorch Extension、HDK和Python版本。

💡 版本配套表在不同模型中的应用策略可能不一致，详情参考[维护策略](#维护策略)。

版本配套表地址：[链接](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)。

## 硬件配套表

昇腾训练设备包含以下型号，都可作为PyTorch模型的训练环境。

硬件配套表：[链接](https://gitee.com/ascend/pytorch#%E7%A1%AC%E4%BB%B6%E9%85%8D%E5%A5%97)


## 维护策略
💡 ModelZoo-pyTorch中的模型区分为随版本演进模型和不随版本演进模型。
- 针对随版本演进模型：请跟随版本配套表，选择最新版本使用。
- 针对不随版本演进模型：这些模型已不随PyTorch和PyTorch Extension的版本演进，您可以选择以下策略。
  - 可根据对应模型的README选择对应PyTorch、PyTorch Extension、CANN、HDK版本使用。
  - 如您对该模型有新版本PyTorch、PyTorch Extension、CANN、HDK的适配需求，可在[modelzoo/issues](https://gitee.com/ascend/modelzoo/issues)提交issue，我们会及时处理。


## 范围界定

### 随版本演进模型

- [albert_ID0335_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/nlp/albert_ID0335_for_PyTorch)
- [Bert_Chinese_ID3433_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/nlp/Bert_Chinese_ID3433_for_PyTorch)
- [BERT-NER-Pytorch(FP32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/nlp/BERT-NER-Pytorch)
- [Bert-Base-Squad_ID0470_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/nlp/Bert-Squad_ID0470_for_PyTorch)
- [Bert-Large-Squad_ID0470_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/nlp/Bert-Squad_ID0470_for_PyTorch)
- [Bert-Large-Squad_ID0470_for_PyTorch(FP32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/nlp/Bert-Squad_ID0470_for_PyTorch)
- [Conformer(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch)
- [CRNN_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/CRNN_for_PyTorch)
- [DB_ID0706_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/detection/DB_ID0706_for_PyTorch)
- [DeeplabV3_for_Pytorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/semantic_segmentation/DeeplabV3_for_Pytorch)
- [EfficientNetV2_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/EfficientNetV2_for_PyTorch)
- [ESPnet2_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/audio/ESPnet2_for_PyTorch)
- [ESPnet2_for_PyTorch(FP32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/audio/ESPnet2_for_PyTorch)
- [Fairseq_Transformer_wmt18_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/nlp/Fairseq_Transformer_wmt18_for_PyTorch)
- [Fairseq_Transformer_wmt18_for_PyTorch(FP32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/nlp/Fairseq_Transformer_wmt18_for_PyTorch)
- [Faster_Mask_RCNN_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/detection/Faster_Mask_RCNN_for_PyTorch)
- [Faster_Mask_RCNN_for_PyTorch(HF32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/detection/Faster_Mask_RCNN_for_PyTorch)
- [HRNet_ID1780_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/classification/HRNet_ID1780_for_PyTorch)
- [HRNet_ID1780_for_PyTorch(HF32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/classification/HRNet_ID1780_for_PyTorch)
- [InceptionV3_ID1596_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/classification/InceptionV3_ID1596_for_PyTorch)
- [Lightweight_OpenPose(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/pose_estimation/Lightweight_OpenPose)
- [MAE_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/classification/MAE_for_PyTorch)
- [MAE_Fine_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/cv/classification/MAE_for_PyTorch)
- [MAPPO_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/rl/MAPPO_for_PyTorch)
- [MobileNetV3-Large_ID1784_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/MobileNetV3-Large_ID1784_for_PyTorch)
- [PPO_for_Pytorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/rl/PPO_for_Pytorch)
- [Resnet50_Cifar_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/Resnet50_Cifar_for_PyTorch)
- [ResNet50_ID4149_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/ResNet50_ID4149_for_PyTorch)
- [ResNet50_ID4149_for_PyTorch(FP32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/ResNet50_ID4149_for_PyTorch)
- [ResNet50_ID4149_for_PyTorch(HF32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/ResNet50_ID4149_for_PyTorch)
- [roberta_for_PyTorch(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/nlp/roberta_for_PyTorch)
- [ShuffleNetV2_iflytek_for_Pytorch(FP32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/dev/perf/ShuffleNetV2_iflytek_for_Pytorch)
- [wav2vec2.0(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/audio/wav2vec2.0)
- [wav2vec2.0(FP32)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/audio/wav2vec2.0)
- [Wenet(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch)
- [whisper(FP16)](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch)
- [Yolov5s_for_PyTorch_v6.0(FP16)](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v6.0)
- [Yolov5s_for_PyTorch_v6.0(FP32)](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v6.0)
- [Yolov5s_for_PyTorch_v6.0(HF32)](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v6.0)
- [Yolov5m_for_PyTorch_v6.0(FP16)](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v6.0)
- [Yolov7_for_PyTorch(FP16)](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/PyTorch/Official/cv/object_detection/Yolov7_for_PyTorch)

### 不随版本演进模型

除以上随版本演进模型范围外，其余ModelZoo-PyTorch中的模型都界定为不随版本演进模型。

## 安全声明

### 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用。

### 文件权限控制

1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控，管控权限可参考表1进行设置。 

    表1 文件（夹）各场景权限管控推荐最大值

    | 类型           | linux权限参考最大值 |
    | -------------- | ---------------  |
    | 用户主目录                        |   750（rwxr-x---）            |
    | 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
    | 程序文件目录                      |   550（r-xr-x---）            |
    | 配置文件                          |  640（rw-r-----）             |
    | 配置文件目录                      |   750（rwxr-x---）            |
    | 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             | 
    | 日志文件(正在记录)                |    640（rw-r-----）           |
    | 日志文件目录                      |   750（rwxr-x---）            |
    | Debug文件                         |  640（rw-r-----）         |
    | Debug文件目录                     |   750（rwxr-x---）  |
    | 临时文件目录                      |   750（rwxr-x---）   |
    | 维护升级文件目录                  |   770（rwxrwx---）    |
    | 业务数据文件                      |   640（rw-r-----）    |
    | 业务数据文件目录                  |   750（rwxr-x---）      |
    | 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
    | 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
    | 加解密接口、加解密脚本            |   500（r-x------）        |


### 运行安全声明

1. 建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。


### 公网地址声明

详见各模型目录下的public_address_statement.md


## 免责声明

### 致ModelZoo使用者
1. ModelZoo提供的模型仅供您用于非商业目的。
2. 对于各模型，ModelZoo平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用ModelZoo模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

### 致数据集所有者
如果您不希望您的数据集在ModelZoo中的模型被提及，或希望更新ModelZoo中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对ModelZoo的理解和贡献。

### License声明
Ascend ModelZoo提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend ModelZoo根目录。