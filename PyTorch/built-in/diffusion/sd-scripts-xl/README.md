# sd-scripts-xl for PyTorch
# 目录
-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [sd-scripts-xl](#sd-scripts-xl)
    - [准备训练环境](#准备训练环境)
    - [准备数据集](#准备数据集)
    - [快速开始](#快速开始)
      - [预训练任务(SDXL+CLIP)](#预训练任务sdxlclip)
      - [预训练任务(SDXL+MT5)](#预训练任务sdxlmt5)
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)



# 简介

## 模型介绍

Stable Diffusion(SD)是计算机视觉领域的一个生成式大模型，能够进行文生图（txt2img）和图生图（img2img）等图像生成任务。sd-scripts仓适配了SD模型的训练、生成以及多个下游任务脚本，包括新版本**Stable Diffusion XL**。目前支持的预训练任务有两个，分别是SDXL+CLIP和SDXL+MT5，两者的区别在于模型的text encoder不一样。
## 支持任务列表

本仓已支持以下模型任务类型。

| 模型        | 任务类型 | 是否支持  |
|-----------| ------- | ------------ |
| SDXL+CLIP | 预训练 | ✅   |
| SDXL+MT5  | 预训练 | ✅   |
## 代码实现
- 参考实现：

  ```
  url=https://github.com/kohya-ss/sd-scripts
  commit_id=46cf41cc93d5856664a2835da2d92796f9344281
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/diffusion/
  ```
# sd-scripts-xl

## 准备训练环境

### 安装模型环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 1**  三方库版本支持表

|   三方库   | 支持版本 |
| :--------: | :------: |
|  PyTorch   |  1.11.0   |
| Accelerate |  0.25.0  |
| Deepspeed  |  0.12.6  |
| diffusers  |  0.21.2  |
| accelerate  |  0.23.0  |
| fairscale  |  0.4.13  |
| torchvision  |  0.12.0  |
| pillow  |  9.5.0  |
| torchvision_npu  |  0.12.0  |
| opencv-python  |  4.6.0.66  |

  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  1. 在模型根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  
     ```
     pip install -r requirements.txt 
     ```
  
  2. 参考[gitee官仓](https://gitee.com/ascend/vision/tree/v0.12.0-dev/)安装对应0.12.0版本的torchvision和torchvision_npu。
  
- 替换三方库补丁。

  在模型根目录下，将以下命令中`python_path`变量赋值为当前环境下的python路径，并执行：

  ```
  python_path=/path/lib/python3.8/site-packages
  \cp $python_path/accelerate/accelerator.py $python_path/accelerate/accelerator.py_bak
  \cp $python_path/accelerate/scheduler.py $python_path/accelerate/scheduler.py_bak
  \cp $python_path/fairscale/optim/oss.py $python_path/fairscale/optim/oss.py_bak
  
  \cp ./third_patch/accelerate_patch/accelerator.py $python_path/accelerate/accelerator.py
  \cp ./third_patch/accelerate_patch/scheduler.py $python_path/accelerate/scheduler.py
  \cp ./third_patch/fairscale_patch/oss.py $python_path/fairscale/optim/oss.py
  ```
### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|      软件类型      | SDXL+CLIP支持版本 | SDXL+MT5支持版本 |
| :----------------: |:-------------:|:-------------:|
| FrameworkPTAdapter |    6.0.RC1    |    6.0.RC1    |
|        CANN        |    8.0.RC1    |    8.0.RC1    |
|    昇腾NPU固件     |   24.1.RC1    |   24.1.RC1    |
|    昇腾NPU驱动     |   24.1.RC1    |   24.1.RC1   |
  


## 准备数据集

1. 用户需自行获取laion数据集，并在shell启动脚本中将`dataset_name`参数，设置为本地数据集的绝对路径，填写一级目录。

   数据结构如下：

   ```
   $dataset
   ├── 000xx.tar
   	├── 000xxxxx.jpg
   	├── 000xxxxx.json
   	└── train-0001.txt
   ├── 000xx.parquet
   └── 000xx_stats.json
   ```
   
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
   
   

## 获取预训练模型

1. 联网情况下，预训练模型会自动下载。

2. 无网络时，用户可访问huggingface官网自行下载，文件namespace如下：

   ```
   stabilityai/stable-diffusion-xl-base-1.0
   madebyollin/sdxl-vae-fp16-fix
   openai/clip-vit-large-patch14
   laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
   ```
3. 如果需要将text_encoder替换为mt5-xxl模型，无网络时，以下文件需要自行下载，文件namespace如下：
   ```
   google/mt5-xxl
   IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
   ```
4. 获取对应的预训练模型后，在shell启动脚本中将`model_name`参数，设置为本地预训练模型路径，填写一级目录。
> **注：**  如果要训练SDXL+MT5模型，需要将模型里面的token length长度从77修改为512，可以使用下面的脚本进行修改，或者参考下面脚本进行手动修改。


```shell
replace_token_length() {
   sed -i 's/77,/512,/g' $1
}
mt5_tokenizer_path=Taiyi-Stable-Diffusion-1B-Chinese-v0.1
replace_token_length $mt5_tokenizer_path/tokenizer_config.json
```



# 快速开始

## 预训练任务(SDXL+CLIP)

本节以预训练为例，展示模型训练方法，其余下游任务txtimg、lora、controlnet、dreambooth、textual inversion等可自行参考适配预训练脚本。

预训练支持单机8卡和多机多卡训练，支持fp16的混合精度训练。
#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机8卡和多机多卡预训练

   - 单机8卡预训练

     ```shell
     bash test/pretrain_full_1m_8p_sdxl.sh # 8卡精度，默认为混精，带FA场景
     bash test/pretrain_full_1m_8p_sdxl.sh --max_train_epoch=13 # 8卡性能，默认为混精，带FA场景
     ```
   
   - 多机多卡预训练
   
     ```shell
     bash test/pretrain_full_nm_np_sdxl.sh --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_ip=x.x.x.x --master_port=8989 # 多卡精度，默认为混精，带FA场景
     bash test/pretrain_full_nm_np_sdxl.sh --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_ip=x.x.x.x --master_port=8989 --max_train_epoch=13 # 多卡性能，默认为混精，带FA场景
     ```
   
     > 脚本参数说明：
     >
     > nnodes：机器数量。
     >
     > nproc_per_node：每台机器的卡数。
     >
     > node_rank：当前机器是几号机器，主机为0，其它为1,2...
     >
     > master_ip：主机ip地址。
     >
     > master_port：主机端口号。
   
   - 模型的python训练脚本参数说明。
   
   ```shell
   sdxl_pretrain.py：
   --max_train_steps                   //最大训练步数
   --max_train_epoch=120               //最大训练轮数
   --pretrained_model_name_or_path     //预训练模型名称或者绝对路径
   --vae                               //vae模型名称或者绝对路径
   --tokenizer1_path                   //tokenizer1名称或者绝对路径
   --tokenizer2_path                   //tokenizer1名称或者绝对路径
   --train_data_dir                    //数据集名称或者绝对路径
   --resolution                        //图片分辨率
   --enable_bucket                     //使能数据集中图片分辨率分桶操作
   --min_bucket_reso                   //分桶操作的最小分辨率
   --max_bucket_reso                   //分桶操作的最大分辨率
   --output_dir                        //输出ckpt的输出路径
   --output_name                       //输出ckpt的前缀
   --save_every_n_epochs               //每n个epoch保存一次权重
   --save_precision                    //保存权重的精度
   --save_model_as                     //保存权重的格式
   --logging_dir                       //输出日志路径
   --gradient_checkpointing            //使能重计算
   --gradient_accumulation_steps       //梯度累计步数
   --learning_rate                     //学习率
   --train_text_encoder                //使能训练text_encoder
   --learning_rate_te1                 //text_encode1的学习率
   --learning_rate_te2                 //text_encode2的学习率
   --lr_warmup_steps                   //学习率预热步数
   --max_grad_norm                     //最大梯度归一值
   --lr_scheduler                      //学习率策略
   --lr_scheduler_num_cycles           //学习率策略中周期数量
   --train_batch_size                  //设置batch_size
   --mixed_precision                   //精度模式
   --seed                              //随机种子
   --caption_extension                 //caption文件的扩展名
   --shuffle_caption                   //打乱caption
   --keep_tokens                       //打乱caption token时保持前N个token不变（token用逗号分隔）
   --optimizer_type                    //优化器类型
   --max_token_length                  //最大token长度
   --enable_npu_flash_attention        //使能npu_flash_attention，仅支持fp16精度
   ```
   
   训练完成后，权重文件保存在`test/output`路径下，并输出模型训练精度和性能信息。
## 预训练任务(SDXL+MT5)
   本节的预训练的模型与上一节预训练的模型的区别在于将模型text_encoder换为MT5模型，本节以预训练为例子，展示模型的训练方法。

   支持单机8卡，bf16混合精度训练。
#### 开始训练
1. 进入解压的源码包根目录。
    ```shell
    cd /${模型文件夹名称}
    ```
2. 运行训练脚本。
      该模型支持单机8卡预训练
      - 单机8卡预训练
      ```shell
      bash test/pretrain_full_8p_sdxl_mt5.sh #8卡训练 默认为混合精度 带FA
      bash test/pretrain_full_8p_sdxl_mt5.sh   --max_train_epoch 1 #8卡性能
      ```
      - 模型的python训练脚本参数说明。
      ```shell
       sdxl_pretrain_mt5.py:
       --pretrained_model_name_or_path               //预训练模型名称或者绝对路径
       --vae                                         //vae模型名称或者绝对路径
       --mt5_tokenizer_path                          //mt5模型tokenizer的路径
       --mt5_encoder_path                            //mt5模型encoder的路径
       --train_data_dir                              //数据集名称或者绝对路径
       --resolution                                  //图片分辨率
       --enable_bucket                               //使能数据集中图片分辨率分桶操作
       --max_data_loader_n_workers                   //数据加载时的多进程数量
       --min_bucket_reso                             //分桶操作的最小分辨率
       --max_bucket_reso                             //分桶操作的最大分辨率
       --output_dir                                  //输出ckpt的输出路径
       --output_name                                 //输出ckpt的前缀
       --logging_dir                                 //输出日志路径
       --max_train_epoch                             //最大训练轮数
       --gradient_checkpointing                      //使能重计算
       --gradient_accumulation_steps                 //梯度累计步数
       --lr_scheduler                                //学习率策略
       --min_snr_gamma                               //用于减少时间步loss的权重的参数
       --learning_rate                               //学习率
       --learning_rate_te1                           //mt5模型的学习率
       --lr_warmup_steps                             //学习率预热步数
       --max_grad_norm                               //最大梯度归一值
       --lr_scheduler_num_cycles                     //学习率策略中周期数量
       --train_batch_size                            //设置batch_size
       --mixed_precision                             //精度模式
       --seed                                        //随机种子
       --caption_extension                           //caption文件的扩展名
       --keep_tokens                                 //打乱caption token时保持前N个token不变
       --optimizer_type                              //优化器类型
       --max_token_length                            //最大token长度
       --no_half_vae                                 //不使用半精度的vae
       --enable_npu_flash_attention                  //使用FA
       --mt5_dim_size                                //mt5模型维度适配
       --bucket_reso_steps                           //Bucket分辨率单位
      ```
      训练完成后，日志文件保存在test/output路径下，并输出模型训练精度和性能信息。
# 训练结果展示

**表 2**  SDXL+CLIP训练结果展示表

|   NAME   | sd版本 | FPS  | batch_size | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :------: | :-----------: | :-----------: |
| GPU | xl | 20.2 | 4 | fp16 |      1.13      |
|  Atlas A2  | xl | 10.4 | 4 | fp16 |      1.11      |


**表 3**  SDXL+MT5预训练训练结果展示表

|   NAME   | sd版本 | FPS  | batch_size | AMP_Type | Torch_Version |
| :------: | :---: | :--: |:----------:|:--------:| :-----------: |
| GPU | xl | 9.754 |     2      |   bf16   |      1.13      |
|  Atlas A2  | xl | 10.71|     2      |   bf16   |      1.11      |





# 公网地址说明
代码涉及公网地址参考 public_address_statement.md


# 变更说明

2023.12.11：首次发布。

2024.03.06：增加SDXL+MT5 模型的预训练
# FAQ


暂无。