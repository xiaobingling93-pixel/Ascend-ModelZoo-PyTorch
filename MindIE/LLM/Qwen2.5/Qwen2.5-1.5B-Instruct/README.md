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
| Qwen2.5-1.5B      | 支持world size 1,2,4,8       | 支持world size 1,2,4,8       | √    | √    | ×               | √               | ×        | ×         | ×            | ×        | ×       | ×              | ×   | ×      | x       | x | x | x |

注：表中所示支持的world size为对话测试可跑通的配置，实际运行时还需考虑输入序列长度带来的显存占用。

- 部署Qwen2.5-1.5B-Instruct模型至少需要1台Atlas 800I A2服务器或者1台插1张Atlas 300I DUO卡的服务器

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

- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main)

## 版本配套
| 模型版本 | transformers版本 |
| -------- | ---------------- |
| Qwen2.5  | 4.43.1           |


## 推理
量化权重生成路径下可能缺少一些必要文件（与转换量化权重时使用的cann版本有关），若启动量化推理失败，请将config.json等相关文件复制到量化权重路径中，可执行以下指令进行复制：
```shell
cp ${浮点权重路径}/*.py ${量化权重路径}
cp ${浮点权重路径}/*.json ${量化权重路径}
cp ${浮点权重路径}/*.tiktoken ${量化权重路径}
```

启动推理时，请在权重路径的config.json文件中添加(或修改)`torch_dtype`字段，例如`"torch_dtype": "bfloat16"`。

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

4.启动qwen需要安装三方依赖tiktoken，若环境中没有该依赖可使用以下命令安装：

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
- 需要先下载数据集（git clone https://modelers.cn/MindIE/data.git），放到/usr/local/Ascend/atb-models/tests/modeltest/下

示例：
```shell
cd /usr/local/Ascend/atb-models/tests/modeltest/
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-1.5B-Instruct权重路径} 2
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
  HCCL_DETERMINISTIC=false LCCL_DETERMINISTIC=0 HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 bash run.sh pa_bf16 performance　[[2048,2048],[1024,1024],[512,512],[256,256]] 1 qwen ${Qwen2.5-1.5B-Instruct权重路径} 2
  ```
注：若权重torch_dtype为float16，则需要修改pa_bf16为 pa_fp16

## FAQ

- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`
- Qwen2.5系列模型当前800I A2采用bf16， 300I DUO使用fp16 

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。