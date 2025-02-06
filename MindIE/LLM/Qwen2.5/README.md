# README

- 千问（Qwen2.5）大语言模型是阿容生成、问答系统等多个场景，助力企业智能化升级。处理能力，能够理解和生成文本，能够应用于智能客服、内容生成、问答系统等多个场景，助力企业智能化升级。

- 此代码仓中实现了一套基于NPU硬件的qwen2.5推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。



# 加载镜像
前往昇腾社区/开发资源(panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml)下载适配Qwen2.5的镜像包：mindie:1.0.0-800I-A2-py311-openeuler24.03-lts、1.0.0-300I-Duo-py311-openeuler24.03-lts

完成之后，请使用docker images命令确认查找具体镜像名称与标签。


# 特性矩阵
- 此矩阵罗列了Qwen2.5模型支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 | prefix_cache | FA3量化 | functioncall | Multi LoRA|
| ----------------- |----------------------------|-----------------------------| ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | -------------- | --- | ------ | ---------- | --- | --- | --- |
| Qwen2.5-0.5B      | 支持world size 1,2,4,8       | 支持world size 1,2,4,8       | √    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | x | x | x |
| Qwen2.5-1.5B      | 支持world size 1,2,4,8       | 支持world size 1,2,4,8       | √    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | x | x | x |
| Qwen2.5-7B        | 支持world size 1,2,4,8       | 支持world size 1,2,4,8       | √    | √    | ×               | √               | √        | ×         | ×            | √        | ×       | ×              | ×   | ×      | √       | x | x | x |
| Qwen2.5-14B       | 支持world size 2,4,8         | 支持world size 1,2,4,8       | √    | √    | ×               | √               | √        | ×         | ×            | √        | ×       | ×              | ×   | ×      | x       | x | x | x |
| Qwen2.5-32B       | 支持world size 4,8           | ×                           | √    | √    | ×               | √               | √        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | x | x | x |
| Qwen2.5-72B       | 支持world size 8             | ×                           | √    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | √ | x | x |

注：表中所示支持的world size为对话测试可跑通的配置，实际运行时还需考虑输入序列长度带来的显存占用。

- 模型支持的张量并行维度(Tensor Parallelism)可以通过查看模型的`config.json`文件中的 **KV头的数量** (`num_key_value_heads` 或者类似字段)来判断模型支持多少维度的张量并行。
> 例如 `Qwen2.5-0.5B` 的 `"num_key_value_heads"` 为 2，表明其只支持world size 1,2 

> 例如 `Qwen2.5-32B` 的 `"num_key_value_heads"` 为 8，表明其理论支持world size 1,2,4,8（不考虑显存占用）

## 路径变量解释

| 变量名      | 含义                                                                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| working_dir     | 加速库及模型库下载后放置的目录                                                                                                                           |
| llm_path        | 模型仓所在路径。若使用编译好的包，则路径为`/usr/local/Ascend/atb-models`；若使用 gitee 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path     | 脚本所在路径；qwen2.5 的工作脚本所在路径为`${llm_path}/examples/models/qwen`                                                                    |
| weight_path     | 模型权重路径                                                                                                                                             |
| rank_table_path | Rank table文件路径                                                                                                                                              |

## 权重

**权重下载**

- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)
- [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/main)
- [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/tree/main)

## 版本配套
| 模型版本 | transformers版本 |
| -------- | ---------------- |
| Qwen2.5  | 4.43.1           |


## 生成量化权重
#### qwen2.5-7b、qwen2.5-14b、qwen2.5-32b W8A8量化、W4A16量化
- W8A8量化权重请使用以下指令生成
    - 当前支持NPU分布式W8A8量化
    - 执行量化脚本
    ```shell
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    注： 安装完cann后 需要执行source set_env.sh 声明ASCEND_HOME_PATH值 后续安装msmodelslim前需保证其不为空
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    cd ${llm_path}
    # 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量 
    # 7b系列使用单卡 14b 32b使用4卡 eg: ASCEND_RT_VISIBLE_DEVICES=4,5,6,7    72B使用8卡 eg: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    vi examples/models/qwen/convert_quant_weight.sh
    # 生成w8a8量化权重
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w8a8
    # 生成w4a16量化权重
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w4a16
    ```

#### qwen2.5-14b、qwen2.5-7b 稀疏量化
- Step 1
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    注： 安装完cann后 需要执行source set_env.sh 声明ASCEND_HOME_PATH值 后续安装msmodelslim前需保证其不为空
    ```shell
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"apt-get update"和"apt install jq"
    jq --version
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    # 指定当前机器上可用的逻辑NPU核心 通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值 指定使用卡号及数量 
    # 7b系列使用单卡 14b 32b使用4卡 eg: ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    vi examples/models/qwen/convert_quant_weight.sh
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w4a8
    ```

- Step 2：量化权重切分及压缩
    ```shell
    export IGNORE_INFER_ERROR=1
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} --multiprocess_num 4
    ```
- TP数为tensor parallel并行个数
- 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
- 示例
    ```shell
    torchrun --nproc_per_node 2 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/Qwen-14b_w8a8s --save_directory /data1/weights/model_slim/Qwen-14b_w8a8sc
    ```

#### Qwen2.5-72B FA3量化
  - 阅读链接中的readme文件生成权重，或者直接问msModelSlim团队索要：
  https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/FA%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md

  - ModelSlim团队会提供`quant_model_description_w8a8.json`和`quant_model_weight_w8a8.safetensors`两个文件。
  - 模型浮点权重中的其他文件（除safetensors文件外）需要手工拷贝到目标量化文件夹中。
  - 拷贝好之后，用户需在`config.json`文件中手动添加以下两个字段：
    ```json
        "quantize": "w8a8",
        "quantization_config": {"fa_quant_type": "FAQuant"}
    ```

## 推理
量化权重生成路径下可能缺少一些必要文件（与转换量化权重时使用的cann版本有关），若启动量化推理失败，请将config.json等相关文件复制到量化权重路径中，可执行以下指令进行复制：
```shell
cp ${浮点权重路径}/*.py ${量化权重路径}
cp ${浮点权重路径}/*.json ${量化权重路径}
cp ${浮点权重路径}/*.tiktoken ${量化权重路径}
```

启动推理时，请在权重路径的config.json文件中添加(或修改)`torch_dtype`字段，例如`"torch_dtype": "float16"`。

启动量化推理时，请在权重路径的config.json文件中添加(或修改)`quantize`字段，值为相应量化方式，例如`"quantize": "w8a8"`、`"quantize": "w8a16"`

在`${llm_path}`目录执行以下指令

```shell
cd /usr/local/Ascend/atb-models
bash examples/models/qwen/run_pa.sh -m ${weight_path} --trust_remote_code true
```

注：

1.推理支持浮点和量化，若启动浮点推理则在`${weight_path}`中传入浮点权重路径，若启动量化则传入量化权重路径

2.--trust_remote_code为可选参数代表是否信任本地的可执行文件，默认false。传入true，则代表信任本地可执行文件，-r为其缩写

3.同时支持Qwen和Qwen1.5模型推理，若启动Qwen模型推理时在`${weight_path}`中传入Qwen路径路径，若启动Qwen1.5模型推理时则在`${weight_path}`中传入Qwen1.5权重路径

4.Qwen系列chat模型需要开启chat模式才能正常输出。
执行：

```shell
cd /usr/local/Ascend/atb-models
bash examples/models/qwen/run_pa.sh -m ${weight_path} --trust_remote_code true -c true
```

5.对于embedding类模型，例如gte-Qwen2-7B-Instruct时，运行命令如下：
```shell
cd /usr/local/Ascend/atb-models
bash examples/models/qwen/run_pa.sh -m ${weight_path} -e true
```

6.启动qwen需要安装三方依赖tiktoken，若环境中没有该依赖可使用以下命令安装：

```shell
pip install tiktoken
```

### run_pa.sh 参数说明（需要到脚本中修改）
根据硬件设备不同请参考下表修改run_pa.sh再运行

| 参数名称                  | 含义                                      | 800I A2推荐值    | 300I DUO推荐值   |
| ------------------------- | ----------------------------------------- | ---------------- | ---------------- |
| BIND_CPU                  | 绑定CPU核心开关,默认进行绑核              | 1                | 1                |
| ASCEND_RT_VISIBLE_DEVICES | 使用的硬件卡号，多个卡间使用逗号相连      | 根据实际情况设置 | 根据实际情况设置 |
| RESERVED_MEMORY_GB        | 保留内存，通常未加速库需要的内存+通信内存 | 3                | 3                |
| MASTER_PORT               | 卡间通信端口,通常不用修改，有冲突时再改   |                  |                  |

注：暂不支持奇数卡并行
    ```

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)

示例：
```shell
cd /usr/local/Ascend/atb-models/tests/modeltest/
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-0.5B-Instruct权重路径} 2
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-1.5B-Instruct权重路径} 2
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-7B-Instruct权重路径} 2
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-14B-Instruct权重路径} 2
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-32B-Instruct权重路径} 8
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-72B-Instruct权重路径} 8
```
注：若权重torch_dtype为float16，则需要修改pa_bf16为 pa_fp16

## 性能测试
- 进入以下路径
  ```shell
  cd /usr/local/Ascend/atb-models/tests/modeltest/
  ```
- 运行指令
  ```shell
  bash run.sh pa_bf16 [performance|full_CEval|full_BoolQ] ([case_pair]) [batch_size] qwen [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])
  ```

- 环境变量释义

1. HCCL_DETERMINISTIC=false          LCCL_DETERMINISTIC=0
这两个会影响性能，开启了变慢，但是会变成确定性计算，不开会变快，所以设置为0。
2. HCCL_BUFFSIZE=120
这个会影响hccl显存，需要设置，基本不影响性能。
3. ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
这个是显存优化，需要开，小batch、短序列场景不开更好。

示例：

  ```shell
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_bf16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen2.5-0.5B-Instruct权重路径} 2
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_bf16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen2.5-1.5B-Instruct权重路径} 2
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_bf16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen2.5-7B-Instruct权重路径} 2
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_bf16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen2.5-14B-Instruct权重路径} 2
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_bf16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen2.5-32B-Instruct权重路径} 8
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_bf16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen2.5-72B-Instruct权重路径} 8
  ```
注：若权重torch_dtype为float16，则需要修改pa_bf16为 pa_fp16

## FAQ

- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`
- Qwen2.5系列模型当前800I A2采用bf16， 300I DUO使用fp16 
