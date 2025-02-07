# Stable-Audio-Tools for PyTorch
# 目录

-   [简介](#简介)
    -  [模型介绍](#模型介绍)
    -  [支持任务列表](#支持任务列表)
    -  [代码实现](#代码实现)
-   [Stable-Audio-2.0](#Stable-Audio-2.0)   
    -   [准备训练环境](#准备训练环境)
    -   [快速开始](#快速开始)
          - [训练任务](#训练任务)
-   [公网地址说明](#公网地址说明) 
-   [变更说明](#变更说明) 
-   [FAQ](#FAQ) 

# 简介
## 模型介绍

Stable-audio-tools是Stability AI 推出的音乐生成平台，给定描述文本提示后生成高质量的音乐和音效。
本仓库主要将Stable-audio模型迁移到了昇腾NPU上。

## 支持任务列表

本仓已经支持以下模型任务类型


|        模型        | 任务列表  | 是否支持 |
|:----------------:|:-----:|:----:|
| stable-audio-2.0 | train |  ✔   |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/Stability-AI/stable-audio-tools
  commit_id=00d35fa90b90aab030b2184c996973165d54fdf9
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/audio/
  ```

# Stable-Audio-2.0

## 准备训练环境

该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
2. 查看[软件版本配套表](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitee.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**

### 安装模型环境

  该模型建议使用python3.9及以上版本，支持的PyTorch历史版本和三方库依赖如下表所示  

  **表 1**  三方库版本支持表

|    三方库     | 支持版本  |
|:----------:|:-----:|
|  PyTorch   | 2.1.0 |
| torchaudio | 2.1.0 |

  在模型源码包根目录下执行命令，安装模型需要的依赖。

  ```
  pip install -e .
  ```

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com /document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                
  
  **表 2**  昇腾软件版本支持表

|        软件类型        |   支持版本   |
|:------------------:|:--------:|
| FrameworkPTAdapter | 6.0.RC2  |
|        CANN        | 8.0.RC2  |
|      昇腾NPU固件       | 24.1.RC2 | 
|      昇腾NPU驱动       |   24.1.RC2   |


### 准备数据集

用户需自行获取ESC-50数据集

参考数据结构如下：

   ```
   /path/to/dataset/ESC-50-master/audio/
   ├── 1-100032-A-0.wav
   ├── 1-100038-A-14.wav
   └── ...
   ```
并修改./stable_audio_tools/configs/dataset_configs/local_training.json中"path"为实际数据集路径
   ```
    "datasets": [
        {
            "id": "my_audio",
            "path": "/path/to/dataset/ESC-50-master/audio/",
            "custom_metadata_module": "./stable_audio_tools/configs/dataset_configs/custom_metadata/custom_md_example.py"
        }
    ],
   ```

### 获取训练相关权重

基于[laion_clap](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt)下载music_audioset_epoch_15_esc_90.14.pt
   
修改./stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json中"clap_ckpt_path"为权重所在路径：

   ```
   "id": "prompt",
    "type": "clap_text",
    "config": {
        "audio_model_type": "HTSAT-base",
        "enable_fusion": true,
        "clap_ckpt_path": "/path/to/music_audioset_epoch_15_esc_90.14.pt",
        "use_text_features": true,
        "feature_layer_ix": -2
    }
   ```


## 快速开始

### 训练任务

#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡训练。
   
  
   - 单机8卡训练
   
     ```shell
     bash test/train_full_8p_stableaudio_2.sh # 8卡训练
     bash test/train_perf_8p_stableaudio_2.sh # 8卡性能
     ```
     
- 训练参数参照目录下defaults.ini文件
   ```shell
   defaults.ini
  
     name                                 \\任务名称
     batch_size                           \\训练batchsize
     num_gpus                             \\训练卡数
     num_nodes                            \\训练节点数 
     strategy                             \\多卡策略
     precision                            \\训练精度
     num_workers                          \\Dataloder使用num_workers数目
     seed                                 \\随机数种子
     accum_batches                        \\梯度累积次数
     checkpoint_every                     \\保存checkpoint频率                             
     ckpt_path                            \\继续训练ckpt路径
     pretrained_ckpt_path                 \\预训练ckpt路径
     pretransform_ckpt_path               \\预处理ckpt路径
     model_config                         \\模型配置文件路径
     dataset_config                       \\数据集配置文件路径
     save_dir                             \\保存路径
     gradient_clip_val                    \\梯度裁剪值
     remove_pretransform_weight_norm      \\去除预处理weight_norm
     max_steps                            \\最大训练步数
   ```
#### 训练结果


##### 性能

|        芯片         | 卡数  | 单步迭代时间（s/step) | batch_size | AMP_Type | Torch_Version |
|:-----------------:|:---:|:--------------:|:----------:|:--------:|:-------------:|
|        竞品A        | 8p  |      1.19      |     4      |   bf16   |      2.1      |
| Atlas 900 A2 PODc | 8p  |      1.44      |     4      |   bf16   |      2.1      |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.07.22：Stable Audio 2.0 bf16训练任务首次发布。

# FAQ
暂无