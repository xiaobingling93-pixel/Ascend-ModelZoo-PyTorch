# Flan-t5-xxl for Pytorch 

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [断点续训](#断点续训)

# 概述

## 简述

Flan-T5是encoder-decoder结构，通过指令在超大规模的任务上进行微调，让语言模型具备了极强的泛化性能，适用于多种NLP任务。

- 参考实现：

  ```
  url=https://github.com/huggingface/transformers.git
  commit_id=345b9b1a6a308a1fa6559251eb33ead2211240ac
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation
  ```


# 准备训练环境

该模型为不随版本演进模型（随版本演进模型范围可在[此处](https://gitcode.com/ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md)查看），未在最新昇腾配套软件中适配验证，您可以：
1. 根据下面提供PyTorch版本在[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中选择匹配的CANN等软件下载使用。
2. 查看[软件版本配套表](https://gitcode.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)后确认对该模型有新版本PyTorch和CANN中的适配需求，请在[modelzoo/issues](https://gitcode.com/ascend/modelzoo/issues)中提出您的需求。**自行适配不保证精度和性能达标。**

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  三方库版本支持表
  
  |     三方库     | 支持版本(PT2.1) | 支持版本(PT2.4) |
  |:-----------:|:-----------:|:-----------:|
  | PyTorch  |     2.1     |     2.4     |
  | transformers |   4.37.2    |   4.37.2    |
   | accelerate |   0.25.0    |   0.25.0    |
   | deepspeed |   0.12.6    |   0.15.3    |
   | datasets |   2.16.0    |   2.16.0    |
   | evaluate |    0.4.1    |    0.4.1    |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

    **表 2**  昇腾软件版本支持表
  
  |     软件类型     |   支持版本   |
  |:-----------:|:--------:|
  | FrameworkPTAdapter  | 24.1.RC1 |
  | CANN |  24.1.RC1  |
   | 昇腾NPU固件 |  24.1.RC1 |
   | 昇腾NPU驱动 |  24.1.RC1  |

  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```shell
  # PyTorch 2.1请使用requirements_2_1.txt
  pip install -r requirements_2_1.txt
  
  # PyTorch 2.4请使用requirements_2_4.txt
  pip install -r requirements_2_4.txt
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。
  

## 准备数据集

1. 下载的数据集squad_v2(train), 数据集结构如下
   ```
   ├── squad_v2
        ├── train-00000-of-00001.parquet
        ├── validation-00000-of-00001.parquet

   ```
2. 下载数据集squad_v2(evaluation)，手动修改squad_v2(evaluation)的数据集名称
   ```shell
   #此处只修改squad_v2(evaluation)数据集，squad_v2(train)数据集保持不变
   mv squad_v2 suqad_v2_eval
   cd squad_v2_eval
   mv squad_v2.py squad_v2_eval.py
   ```
   修改后squad_v2_eval数据集结构如下
   ```
   ├── squad_v2_eval
        ├── app.py
        ├── compute_score.py
        ├── requirements.txt
        ├── squad_v2_eval.py
        ├── README.md
   ```

## 准备预训练权重
下载的权重google/flan-t5-xxl, 模型权重的结构如下:
   ```
   ├── flan-t5-xxl
        ├── config.json
        ├── generation_config.json
        ├── special_tokens_map.json
        ├── spiece.model
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── model.safetensors.index.json
        ├── model-00001-of-00005.safetensors
        ├── model-00002-of-00005.safetensors
        ...
        ├── model-00005-of-00005.safetensors
   ```

# 开始训练

## 训练模型
 
1. 将数据集，权重，移动到源码所在路径
   ```shell
   # 将下载的权重配置移动到对应路径
   mv google/flan-t5-xxl PyTorch/built-in/foundation/Flan-t5-xxl
   # 将下载的数据集移动到对应路径
   mv squad_v2 PyTorch/built-in/foundation/Flan-t5-xxl
   mv squad_v2_eval PyTorch/built-in/foundation/Flan-t5-xxl
   ```
   
   完成文件移动后，实际训练环境文件目录结构如下
   ```
   ├── PyTorch/built-in/foundation/Flan-t5-xxl 
        ├── google
            ├── flan-t5-xxl
        ├── scripts
            ├── env_npu.sh   
            ├── train_full_8p.sh
        ├── utils
            ├── trainer_seq2seq_qa.py 
            ├── run_seq2seq_qa.py 
            ├── rms_norm.py 
        ├── squad_v2
        ├── squad_v2_eval
        ├── ds_config.json
   ```

2. 修改训练脚本参数
    - 进入脚本所在目录
    ```shell
    cd PyTorch/built-in/foundation/Flan-t5-xxl/scripts
    ```
   - 模型训练脚本参数说明如下。

   ```
    --model_name_or_path          //模型权重路径
    --deepspeed                   //deepspeed配置文件路径
    --dataset_name                //squad_v2路径 
    --eval_dataset_name           //评估数据集路径
    --version_2_with_negative     //squad_v2数据集处理参数，用于处理question-answer数据结构无answer情况
    --context_column              //数据集context_column 
    --question_column             //数据集question_column 
    --answer_column               //数据集answer_column
    --do_train                    //训练标识
    --do_eval                     //评估标识
    --per_device_train_batch_size //训练时单卡批次大小
    --learning_rate 5e-6          //初始学习率
    --gradient_accumulation_steps //梯度累计步数 
    --num_train_epochs            //重复训练次数
    --max_steps 2000              //训练最大步数
    --save_steps 2000             //保存checkpoint步数间隔
    --max_seq_length              //seq_length 
    --logging_steps 1             //日志打印步数
    --doc_stride 128              //滑动步长
    --seed 1234                   //随机种子
    --bf16                        //混精bf16
    --overwrite_output_dir        //覆盖输出目录
   ```
   - 修改`model_name_or_path`,`deepspeed`,`dataset_name`, `eval_dataset_name`为绝对路径位置
    
   
3. 运行训练脚本，该模型支持单机8卡训练。
     
    ```shell
    bash train_full_8p.sh # 8卡精度及性能
    ```
   训练完成后，权重文件保存在`PyTorch/built-in/foundation/Flan-t5-xxl/output`路径下，模型训练精度和性能信息保存在`PyTorch/built-in/foundation/Flan-t5-xxl/scripts/output`文件夹下。

4. 竞品运行修改参数

   - 修改`utils/run_seq2seq_qa.py`文件, 注释代码
      ```python
      #from rms_norm import forward
      #T5LayerNorm.forward = forward
      ```
   - 修改`scripts/train_full_8p.sh`文件，注释代码
      ```shell
      #source ${scripts_path_dir}/env_npu.sh
      ```
# 训练结果展示

  **表 3**  性能结果展示表(2000 steps)

|  Name  | Train Samples Per Second | Iterations | DataType | Torch_Version |
|:------:|:------------------------:|:----:|:--------:|:-------------:|
| 8p-NPU |          17.597          |   2000   |     BF16      |     2.1.0      |
| 8p-竞品A |          22.012          |   2000   |     BF16      |     2.1.0     |

  **表 4** eval 结果展示表(2000 steps)

|  Name  | eval loss | 
|:------:|:---------:|
| 8p-NPU |  0.3771   |
| 8p-竞品A | 0.3792    | 


# 断点续训
断点续训是huggingface transformers仓库自带的功能，需要使用断点续训只需要在shell脚本中添加对应的参数即可。

- 保存checkpoint，需要在`train_full_8p.sh`里添加参数`save_steps`，代表每隔多少步保存一次checkpoint

    ```
    --save_steps 50
    ```

- 上述参数添加后，训练完毕，脚本会在`PyTorch/built-in/foundation/Flan-t5-xxl/output`下生成checkpoint文件
- 加载checkpoint，需要在`train_full_8p.sh`脚本里添加参数，其中`$PATH`在此流程下，一般指代的路径为`PyTorch/built-in/foundation/Flan-t5-xxl/output`下的checkpoint文件，例如tmp-checkpoint-80
    ```shell
    --resume_from_checkpoint $PATH
    ```

重新执行训练，就会从checkpoint之后的一步开始训练

# 版本说明

### 变更
2024.01.31: 首次发布
2024.02.26: 修正训练脚本

### FAQ
无