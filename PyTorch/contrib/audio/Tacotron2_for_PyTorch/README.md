# Tacotron2 for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)


# 概述

## 简述

Tacotron2是一个从文字直接转化为语音的神经网络。这个体系是由字符嵌入到梅尔频谱图的循环序列到序列神经网络组成的，然后是经过一个修改过后的WaveNet，该模型的作用是将频谱图合成波形图。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/
  commit_id=9a6c5241d76de232bc221825f958284dc84e6e35  
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/audio
  ```


# 准备训练环境

该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**

## 准备环境

- 当前模型支持的 PyTorch 历史版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.8 | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令。
  ```
  pip install -r requirements.txt
  ```
  >**说明：**
  >LLVM版本与numbra、llvmlite版本号严格依赖，如LLVM 7.0对应llvmlite的0.30.0，numbra的0.46.0版本。



## 准备数据集

1. 获取数据集。

   用户自行下载 `LJSpeech-1.1` 数据集，上传到源码包根目录下并解压，然后在源码包根目录下运行scripts/prepare_mels.sh。
   ```
   bash scripts/prepare_mels.sh    
   ```
   数据集目录结构参考如下所示。
   ```
   ├──LJSpeech-1.1
       ├── mels            
       ├── metadata.csv            
       ├── README
       └── wavs           
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=./LJSpeech-1.1  # 单卡精度

     bash ./test/train_performance_1p.sh --data_path=./LJSpeech-1.1 # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=./LJSpeech-1.1  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=./LJSpeech-1.1 # 8卡性能
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   -m                                  //训练模型名称
   -o                                  //训练文件输出路径  
   --amp                               //是否使用apex混合精度训练
   --epochs                            //重复训练次数
   --bs                                //训练批次大小
   --lr                                //初始学习率
   --seed                              //随机种子
   --weight-decay                      //权重衰减系数
   --training-files                    //训练文件
   --validation-files                  //验证文件
   ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Accuracy | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |     |      | 1        | O2       | 1.8 |
| 8p-竞品A |     |  236913  | 301      | O2       | 1.8 |
| 1p-NPU |     |      | 1        | O2       | 1.8 |
| 8p-NPU |     | 69470 | 301      | O2       | 1.8 |


# 版本说明:

## 变更

2023.1.12：整改Readme，重新发布。

## FAQ

无。

# 公网地址说明

代码涉及公网地址参考 [public_address_statement.md](./public_address_statement.md)