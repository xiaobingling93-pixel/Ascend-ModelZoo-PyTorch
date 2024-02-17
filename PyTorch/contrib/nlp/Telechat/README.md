# telechat模型-训练指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [训练环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [基础环境搭建](#section4622531142816)
  - [模型全参微调](#section741711594517)

- [推理样例](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  - 星辰语义大模型TeleChat是由中电信人工智能科技有限公司研发训练的大语言模型，采用1.5万亿 Tokens中英文高质量语料进行训练。
  - 本模型为对话模型**TeleChat-7B-bot**，提供了`huggingface`格式的权重文件。
  
  - 参考实现：
    ```填写github链接
    https://github.com/Tele-AI/Telechat
    ```
  - 适配昇腾 AI 处理器的实现：
    ```
    https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/contrib/nlp/Telechat
    ```
  
# 训练环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  
  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.0  | [固件与驱动](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/258924109?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
  | CANN（toolkit+kernels）                                     | 7.0.0   | [CANN](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/258923273?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
  | FrameworkPTAdapter (pytorch2.1.0)                           | 5.0.0   | [PTA](https://gitee.com/ascend/pytorch/releases/) | 
  | Python                                                     | 3.9   | -                                                            |            


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 基础环境搭建<a name="section4622531142816"></a>

1. 训练环境搭建

   ```bash
   # 下载镜像
   wget https://telechat-docker.obs.myhuaweicloud.com/docker/telechat_train.tar.gz
   # 读取镜像
   docker load < telechat_train.tar.gz
   # 拉取telechat开源代码仓
   git clone https://github.com/Tele-AI/Telechat
   # 拉取昇腾适配代码仓
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   ```

2. 昇腾软件环境搭建

   ```
   # 安装驱动
   chmod +x ./Ascend-hdk-910b-npu-driver_23.0.0_linux-aarch64.run
   ./Ascend-hdk-910b-npu-driver_23.0.0_linux-aarch64.run --full
   # 安装固件
   chmod +x ./Ascend-hdk-910b-npu-firmware_7.1.0.3.220.run
   ./Ascend-hdk-910b-npu-firmware_7.1.0.3.220.run --full
   ```

3. 数据集下载
   ```
   wget https://telechat-docker.obs.cn-north-4.myhuaweicloud.com/example_dataset.jsonl
   ```

## 模型全参微调训练<a name="section741711594517"></a>

1. 获取预训练权重

   在当前文件夹下载预训练权重，执行以下命令。Huggingface权重下载需要配置git-lfs

   ```bash
   # 查看服务器操作系统，根据操作系统选择git-lfs下载方式
   uname -a
   # 下载 git-lfs ubuntu版
   sudo apt-get install git-lfs
   # 下载 git-lfs centos版
   sudo yum install git-lfs
   # 初始化git-lfs
   git lfs install
   # 下载预训练权重
   git clone https://huggingface.co/Tele-AI/Telechat-7B
   ```

2. 启动容器

   修改telechat_docker_start.sh脚本中第23行冒号前后路径为实际代码所在文件夹路径
   ```bash
   bash telechat_docker_start.sh
   ```
   
3. （**可选**）下载安装cann-toolkit/cann-kernels

   镜像中已经包含了cann-toolkit与cann-kernels
   ```
   # 安装toolkit
   chmod +x ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run
   ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --full
   chmod +x ./Ascend-cann-kernels-910b_7.0.0_linux.run
   /Ascend-cann-kernels-910b_7.0.0_linux.run --install
   ```
   
4. 安装环境依赖，您可以通过安装以下依赖在裸机部署Telechat。
   
  
   4.1 (**可选**)安装 torch 和 torch_npu   
   ```
   pip install torch-2.1.0-cp39-cp39m-manylinux2014_aarch64.whl
   pip install torch_npu-2.1.0*-cp39-cp39m-linux_aarch64.whl 
   ```
   4.2  (**必选**)安装 megatron-core
   ```
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM 
   git checkout 23.05
   pip install -e . 
   ```
   4.3  (**可选**) 安装 deepspeed和 (**必选**) deepspeed_npu 
   ```
   pip3 install deepspeed==0.9.2 
   git clone https://gitee.com/ascend/DeepSpeed.git -b v0.9.2 deepspeed_npu
   cd deepspeed_npu
   pip3 install -e ./
   cd ..
   ```
   4.4 (**可选**)安装python依赖  
   ```
   pip install -r requirements.txt 
   ```

5. 代码适配改动

   将代码仓Telechat/deepspeed-telechat/sft/main.py替换为ModelZoo中提供的main.py
   ```
   mv ./Telechat/deepspeed-telechat/sft/main.py ./Telechat/deepspeed-telechat/sft/main.py.bak
   cp ./ModelZoo-PyTorch/PyTorch/contrib/nlp/Telechat/main.py ./Telechat/deepspeed-telechat/sft/main.py
   ```
   
   将代码仓Telechat/models/7B/modeling_telechat.py替换为ModelZoo中提供的modeling_telechat.py
   ```
   mv ./Telechat/models/7B/modeling_telechat.py ./Telechat/models/7B/modeling_telechat.py.bak
   cp ./ModelZoo-PyTorch/PyTorch/contrib/nlp/Telechat/modeling_telechat.py ./Telechat/models/7B/modeling_telechat.py
   ```
   
6. 开始训练。
   
   该模型支持单机单卡训练和单机8卡训练。
   
   - 单机单卡训练

     将run_telechat_single_node.sh中的ZERO_STAGE调整为1或2，启动单卡训练。

     ```
     cd deepspeed-telechat/sft
     # 配置run_telechat_single_node.sh中data_path、model_name_or_path参数执行训练。
     bash run_telechat_single_node.sh
     ```

   - 单机8卡训练

     将run_telechat_single_node.sh中的ZERO_STAGE调整为1或2，启动8卡训练。

     ```
     cd deepspeed-telechat/sft
     # 配置run_telechat_multi_node.sh中data_path、model_name_or_path参数执行训练。
     bash run_telechat_multi_node.sh
     ```

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                              //数据集路径
   --model_name_or_path                     //模型路径
   --per_device_train_batch_size            //每张卡训练的bs
   --max_seq_len                            //最大输入长度      
   --learning_rate                          //初始学习率
   --num_train_epochs                       //训练批次大小
   --lr_scheduler_type                      //学习率策略
   --weight_decay                           //权重衰减
   --output_dir                             //模型输出路径
   --ZERO_STAGE                             //显存优化策略

   训练完成后，权重文件保存在--output_dir下，并输出模型性能信息。
   
6. 训练结果展示

    **表 2**  训练结果展示表
    
    | NAME    |  performance(samples/s) | Epochs | AMP_Type |
    | ------- | ---: | ------ | -------: |
    | 8p-NPU  |  8.8 | 5    |       O2 |
    
    通过对比训练Loss下降对比精度
    npu 以及gpu 的训练loss 均从2.7下降至1.0左右；
    npu 与gpu的最大误差≤0.02；(见下图)
    
    ![image](https://gitee.com/wangzki/images/raw/master/1.png)

# 推理样例<a name="ZH-CN_TOPIC_0000001172201573"></a>
   ```
   # 拷贝该目录下的configuration_telechat.py，generation_config.json，generation_utils.py，modeling_telechat.py至训练完成的目录下
   # 修改 config.json 中的flash_attn为false 
   # 启动推理：
   python3 telechat_infer_demo.py models_path
   ```

   - 参数说明：
      --models_path：传入训练命令中设置的output_dir路径，即完成训练的模型路径

   备注说明：长句出现报错时，可设置max_length限制输出大小；流式返回及续写当前暂不支持。