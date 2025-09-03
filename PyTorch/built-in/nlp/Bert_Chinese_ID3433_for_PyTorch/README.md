# Bert_Chinese for PyTorch

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)


# 概述

## 简述

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，是一种用于自然语言处理（NLP）的预训练技术。Bert-base模型是一个12层，768维，12个自注意头（self attention head）,110M参数的神经网络结构，它的整体框架是由多层transformer的编码器堆叠而成的。

- 参考实现：

  ```
  url=https://github.com/huggingface/transformers
  commit_id=d1d3ac94033b6ea1702b203dcd74beab68d42d83
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```


# 准备训练环境

## 准备环境

- 推荐使用最新的版本准备训练环境。

  **表 1**  版本配套表
    
  <table border="0">
    <tr>
      <th>软件</th>
      <th>版本</th>
      <th>安装指南</th>
    </tr>
    <tr>
      <td> Driver </td>
      <td> AscendHDK 25.0.RC1.1 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0005.html">驱动固件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> Firmware </td>
      <td> AscendHDK 25.0.RC1.1 </td>
    </tr>
    <tr>
      <td> CANN </td>
      <td> CANN 8.1.RC1 </td>
      <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
    </tr>
    <tr>
      <td> PyTorch </td>
      <td> 2.1.0 </td>
      <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
    </tr>
    <tr>
      <td> torch_npu </td>
      <td> release v7.0.0-pytorch2.1.0 </td>
    </tr>
  </table>
  
- 安装依赖：

  在模型根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```

- 安装transformers：

  ```
  cd transformers
  pip3 install -e ./
  cd ..
  ```

## 准备数据集

1. 获取数据集。

    用户自行下载 `zhwiki` 数据集。

    解压得到zhwiki-latest-pages-articles.xml。

    ```
    bzip2 -dk zhwiki-latest-pages-articles.xml.bz2
    ```

    使用模型根目录下的WikiExtractor.py提取文本，其中extracted/wiki_zh为保存路径，不要修改。

    ```
    python3 WikiExtractor.py zhwiki-latest-pages-articles.xml -b 100M -o extracted/wiki_zh
    ```

    将多个文档整合为一个txt文件，在模型根目录下执行。

    ```
    python3 WikicorpusTextFormatting.py --extracted_files_path extracted/wiki_zh --output_file zhwiki-latest-pages-articles.txt
    ```

    最终生成的文件名为zhwiki-latest-pages-articles.txt。

    Bert-base下载配置模型和分词文件。

    ```
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
    ```

    将下载下的bert-base-chinese放置在模型根目录下。

    Bert-large下载配置模型和分词文件。

    ```
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/algolet/bert-large-chinese
    ```

    将下载下的bert-large-chinese放置在模型根目录下。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡训练以及双机多卡训练。

   - 单机单卡训练

     启动base单卡训练。

     ```
     bash test/train_full_1p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --device_id=0  # 单卡精度训练
     bash test/train_performance_1p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 单卡性能训练   
     ```
     启动large单卡训练。

     ```
     bash test/train_full_1p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --device_id=0 --warmup_ratio=0.1 --weight_decay=0.00001  # 单卡精度训练
     bash test/train_performance_1p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --warmup_ratio=0.1 --weight_decay=0.00001    # 单卡性能训练   
     ```

   - 单机8卡训练

     启动base8卡训练。

     ```
     bash test/train_full_8p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base     # 8卡精度训练
     bash test/train_performance_8p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 8卡性能训练  
     ```
     启动large8卡训练。

     ```
     bash test/train_full_8p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --warmup_ratio=0.1 --weight_decay=0.00001   # 8卡精度训练
     bash test/train_performance_8p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --warmup_ratio=0.1 --weight_decay=0.00001   # 8卡性能训练  
     ```

   - 单机16卡训练

     启动base16卡训练。

     ```
     bash test/train_full_16p_single_node.sh --data_path=dataset_file_path --batch_size=32 --model_size=base     # 16卡精度训练
     bash test/train_performance_16p_single_node.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 16卡性能训练  
     ```

   - 多机多卡训练
   
     启动base多机多卡训练。

     ```
     bash test/train_full_multinodes.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx  # 多机多卡精度训练
     bash test/train_performance_multinodes.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx  #多机多卡性能训练
     ```
     
     启动large多机多卡训练。

     ```
     bash test/train_full_multinodes.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx --warmup_ratio=0.1 --weight_decay=0.00001 # 多机多卡精度训练
     bash test/train_performance_multinodes --data_path=dataset_file_path --batch_size=16 --model_size=large --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx --warmup_ratio=0.1 --weight_decay=0.00001 # 多机多卡性能训练
     ```

     ```
       --data_path：  数据集路径
       --device_number: 每台服务器上要使用的训练卡数
       --model_size： 训练model是base或者是large
       --device_id：  单卡训练时所使用的device_id
       --node_rank:   集群节点序号，master节点是0， 其余节点依次加1
       --master_addr：master节点服务器的ip
       --master_port: 分布式训练中,master节点使用的端口
     ```
   
   - 双机8卡训练  
     启动双机8卡训练。

     ```
     bash ./test/train_cluster_8p.sh --data_path=real_data_path --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx 
     ```
     
     ```
     --node_rank                              //集群节点序号，master节点是0，其余节点依次加1
     --master_addr                            //master节点服务器的ip
     --master_port                            //分布式训练中，master节点使用的端口
     --data_path                              //数据集路径,需写到数据集的一级目录。
     ```
   
   模型训练脚本参数说明如下。

    ```
    公共参数：
    --config_name                            //模型配置文件
    --model_type                             //模型类型
    --tokenizer_name                         //分词文件路径
    --train_file                             //数据集路径
    --eval_metric_path                       //精度评估处理脚本路径
    --line_by_line                           //是否将数据中一行视为一句话
    --pad_to_max_length                      //是否对数据做padding处理
    --remove_unused_columns                  //是否移除不可用的字段
    --save_steps                             //保存的step间隔
    --overwrite_output_dir                   //是否进行覆盖输出
    --per_device_train_batch_size            //每个卡的train的batch_size
    --per_device_eval_batch_size             //每个卡的evaluate的batch_size
    --do_train                               //是否进行train
    --do_eval                                //是否进行evaluate
    --fp16                                   //是否使用混合精度
    --fp16_opt_level                         //混合精度level
    --loss_scale                             //loss scale值
    --use_combine_grad                       //是否开启tensor叠加优化
    --optim                                  //优化器
    --output_dir                             //输出保存路径
    ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

3. 在线推理  
   - 启动在线推理。
 
    ```
    bash ./test/train_eval_1p.sh --data_path=real_data_path --device_id=xxx --checkpoint=real_checkpoint_path
    ```

    ```
    --data_path： 数据集路径
    --device_id：  在线推理时所使用的device_id
    --checkpoint:  权重文件目录
    ```

# 训练结果展示

**表2**  训练结果展示表

|   NAME   | Acc@1 |   FPS    | Epochs | AMP_Type | Torch_Version |
| :------: | :---: |:--------:| :----: | :------: | :-----------: |
| 1p-竞品V |  - | - |  3   |    O2     |      1.5      |
| 8p-竞品V |  0.59 | 898 |  3   |    O2     |      1.5      |
| 1p-NPU | - | 171.644  |   3    |     O2     |     1.11      |
| 8p-NPU | 0.59 | 1352.878 |   3    |     O2     |     1.11      |
| 1p-NPU | - | 171.687  |   3    |     O2     |      2.1      |
| 8p-NPU | 0.59 | 1357.106 |   3    |     O2     |      2.1      |

说明：上表为历史数据，仅供参考。2025年5月10日更新的性能数据如下：
| NAME | 精度类型 | FPS |
| :------: |:-------:|:------:|
| 8p-竞品 | FP16 | 898 |
| 8p-Atlas 900 A2 PoDc | FP16 | 1573.49 |

# 版本说明

## 变更

2024.06.28: 新增单机16卡脚本，增加16卡性能基线。

2022.08.24：首次发布

## FAQ

1. Q:第一次运行报类似"xxx **socket timeout** xxx"的错误该怎么办？

   A:第一次运行tokenizer会对单词进行预处理，根据您的数据集大小，耗时不同，若时间过长，可能导致等待超时。此时可以通过设置较大的超时时间阈值尝试解决：

    （1）设置pytorch框架内置超时时间，修改脚本中的distributed_process_group_timeout（单位秒）为更大的值，例如设置为7200：
   
    ```
    --distributed_process_group_timeout 7200
    ```

    （2）设置HCCL的建链时间为更大的值，修改env.sh中环境变量HCCL_CONNECT_TIMEOUT（单位秒）的值：

    ```
    export HCCL_CONNECT_TIMEOUT=7200
    ```
2. Q:如果训练报wandb.error.UsageError:api_key not configured (no-tty)的错误该怎么办?
  
   A:export WANDB_DISABLED=1


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
