# 简介

- 千问（Qwen2.5）大语言模型能够理解和生成文本，应用于智能客服、内容生成、问答系统等多个场景，助力企业智能化升级。

- 此代码仓中实现了一套基于NPU硬件的Qwen2.5推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。


# 特性矩阵
- 此矩阵罗列了Qwen2.5模型支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 | prefix_cache | FA3量化 | functioncall | Multi LoRA|
| ----------------- |----------------------------|-----------------------------| ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | -------------- | --- | ------ | ---------- | --- | --- | --- |
| Qwen2.5-7B      | 支持world size 1,2,4,8       | 支持world size 1,2,4,8       | √    | √(800I A2/32G/64G)    | ×               | √               | √(800I A2/32G/64G)        | ×        | ×            | √(300I DUO)        | ×       | √              | ×   | ×      | √       | × | √ | x |

注：表中所示支持的world size为对话测试可跑通的配置，实际运行时还需考虑输入序列长度带来的显存占用。

- 部署Qwen2.5-7B-Instruct模型至少需要1台Atlas 800I A2服务器或者1台配置一张Atlas 300I DUO推理卡的服务器。

## 路径变量解释

| 变量名      | 含义                                                                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| working_dir     | 加速库及模型库下载后放置的目录                                                                                                                           |
| llm_path        | 模型仓所在路径。若使用编译好的包，则路径为`/usr/local/Ascend/atb-models`；若使用 gitee 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path     | 脚本所在路径；Qwen2.5 的工作脚本所在路径为`${llm_path}/examples/models/qwen`                                                                    |
| weight_path     | 模型权重路径                                                                                                                                             |
| rank_table_path | Rank table文件路径                                                                                                                                              |

## 加载镜像
前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载`1.0.0-300I-Duo-py311-openeuler24.03-lts`或`1.0.0-800I-A2-py311-openeuler24.03-lts`镜像，下载镜像前需要申请权限，耐心等待权限申请通过后，根据指南下载对应镜像文件。

完成之后，请使用docker images命令确认查找具体镜像名称与标签。

## 启动容器
- 执行以下命令启动容器（参考）：
```sh
docker run -itd --privileged  --name= {容器名称}  --net=host \
   --shm-size 500g \
   --device=/dev/davinci0 \
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   --device=/dev/davinci3 \
   --device=/dev/davinci4 \
   --device=/dev/davinci5 \
   --device=/dev/davinci6 \
   --device=/dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v  {/权重路径:/权重路径}  \
   -v  {swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-XXX-800I-A2-arm64-py3.11（根据加载的镜像名称修改）}  \
   bash
```
#### 进入容器
- 执行以下命令进入容器（参考）：
```sh
docker exec -it {容器名称} bash
```

#### 设置基础环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
```

## 模型权重

**权重下载**

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)

## 版本配套
| 模型版本 | transformers版本 |
| -------- | ---------------- |
| Qwen2.5  | 4.46.3           |


## 生成量化权重
#### qwen2.5-7b W8A8量化
- W8A8量化权重请使用以下指令生成
    - 当前支持NPU分布式W8A8量化
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    - 执行量化脚本
    ```shell
    注： 安装完cann后，需要执行source set_env.sh，声明ASCEND_HOME_PATH值，后续安装msmodelslim前需保证其不为空。
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"yum update"和"yum install jq"
    jq --version
    cd ${llm_path}
    # 指定当前机器上可用的逻辑NPU核心，通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值，指定使用卡号及数量。
    # 7b系列使用单卡  eg: ASCEND_RT_VISIBLE_DEVICES=0
    vi examples/models/qwen/convert_quant_weight.sh
    # 生成w8a8量化权重
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w8a8
    ```

#### qwen2.5-7b 稀疏量化 (仅300I DUO卡支持)
- Step 1
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
    ```shell
    注： 安装完cann后，需要执行source set_env.sh，声明ASCEND_HOME_PATH值，后续安装msmodelslim前需保证其不为空.
    # 执行"jq --version"查看是否安装jq，若返回"bash：jq：command not found"，则依次执行"yum update"和"yum install jq"
    jq --version
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    # 指定当前机器上可用的逻辑NPU核心，通过修改convert_quant_weight.sh文件中export ASCEND_RT_VISIBLE_DEVICES值，指定使用卡号及数量 
    # 7b系列使用单卡 eg: ASCEND_RT_VISIBLE_DEVICES=0
    vi examples/models/qwen/convert_quant_weight.sh
    bash examples/models/qwen/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type qwen_w4a8
    ```

- Step 2：量化权重切分及压缩
    ```shell
    export IGNORE_INFER_ERROR=1
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} --multiprocess_num 4
    ```
    TP数为tensor parallel并行个数
    > 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
    
    示例:
    ```shell
    torchrun --nproc_per_node 2 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/Qwen-7b_w8a8s --save_directory /data1/weights/model_slim/Qwen-7b_w8a8sc
    ```

## 纯模型推理：
【使用场景】使用相同输入长度和相同输出长度，构造多Batch去测试纯模型性能


#### 参数说明（需要到脚本中修改）
根据硬件设备不同请参考下表修改`/usr/local/Ascend/atb-models/examples/models/qwen/run_pa.sh`再运行

| 参数名称                  | 含义                                      | 800I A2推荐值    | 300I DUO推荐值   |
| ------------------------- | ----------------------------------------- | ---------------- | ---------------- |
| BIND_CPU                  | 绑定CPU核心开关,默认进行绑核              | 1                | 1                |
| ASCEND_RT_VISIBLE_DEVICES | 使用的硬件卡号，多个卡间使用逗号相连      | 根据实际情况设置 | 根据实际情况设置 |
| RESERVED_MEMORY_GB        | 保留内存，通常未加速库需要的内存+通信内存 | 3                | 3                |
| MASTER_PORT               | 卡间通信端口,通常不用修改，有冲突时再改   |       /        |         /        |

> 注：暂不支持奇数卡并行

- 环境变量释义

```
export HCCL_DETERMINISTIC=false          
export LCCL_DETERMINISTIC=0
export HCCL_BUFFSIZE=120
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
```
1. `HCCL_DETERMINISTIC`和`LCCL_DETERMINISTIC`这两个会影响性能，开启了变慢，但是会变成确定性计算，不开会变快，所以设置为0。
2. `HCCL_BUFFSIZE`会影响hccl显存，需要设置，基本不影响性能。
3. `ATB_WORKSPACE_MEM_ALLOC_GLOBAL`是显存优化，需要开，小batch、短序列场景不开更好。


#### 对话测试
1. 进入atb-models路径
```
cd /usr/local/Ascend/atb-models
```
2. 清理残余进程：
```
pkill -9 -f 'mindie|python'
```
3. 执行命令：
```
export MINDIE_LOG_TO_STDOUT=1
bash examples/models/qwen/run_pa.sh -m ${weight_path} --trust_remote_code true
```


#### 精度测试
1. 进入modeltest路径
```
cd /usr/local/Ascend/atb-models/tests/modeltest/
```

2. 清理残余进程：
```
pkill -9 -f 'mindie|python'
```

3. 执行命令：
```
export MINDIE_LOG_TO_STDOUT=1
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_model]) [weight_dir] [world_size]
```
参数说明：
1. `data_type`：为数据类型，根据权重目录下config.json的data_type选择bf16或者fp16，例如：pa_bf16。
2. `dataset`：可选full_BoolQ、full_CEval等，相关数据集需要自行下载，（参考[附录]，下载之后拷贝到/usr/local/Ascend/atb-models/tests/modeltest/路径下）CEval与MMLU等数据集需要设置`shots`（通常设为5）。
3. `batch_size`：为`batch数`。
4. `model_name`：为`qwen`。
5. `is_chat_model`：为`是否支持对话模式，若传入此参数，则进入对话模式`。
6. `weight_dir`：为模型权重路径。
7. `world_size`：为总卡数。


样例 -BoolQ
```
bash run.sh pa_bf16 full_BoolQ 1 qwen ${Qwen2.5-7B-Instruct权重路径} 8
```

样例 -CEval
```
bash run.sh pa_bf16 full_CEval 5 1 qwen ${Qwen2.5-7B-Instruct权重路径} 8
```


#### 性能测试
1. 进入modeltest路径：
```
cd /usr/local/Ascend/atb-models/tests/modeltest/
```
2. 清理残余进程：
```
pkill -9 -f 'mindie|python'
```
3. 执行命令：
```
export MINDIE_LOG_TO_STDOUT=1
bash run.sh pa_[data_type] performance [case_pair] [batch_size] ([prefill_batch_size]) [model_name] ([is_chat_model]) [weight_dir] [world_size]
```
参数说明：
1. `data_type`：为数据类型，根据权重目录下config.json的data_type选择bf16或者fp16，例如：pa_bf16。
2. `case_pair`：[最大输入长度,最大输出长度]。
3. `batch_size`：为`batch数`。
4. `prefill_batch_size`：为可选参数，设置后会固定prefill的batch size。
5. `model_name`：为`qwen`。
6. `is_chat_model`：为`是否支持对话模式，若传入此参数，则进入对话模式`。
7. `weight_dir`：为模型权重路径。
8. `world_size`：为总卡数。

样例：
```
bash run.sh pa_bf16 performance [[256,256]] 1 qwen ${Qwen2.5-7B-Instruct权重路径} 8
```


## 服务化推理：
【使用场景】对标真实客户上线场景，使用不同并发、不同发送频率、不同输入长度和输出长度分布，去测试服务化性能
#### 配置服务化环境变量

变量含义：expandable_segments-使能内存池扩展段功能，即虚拟内存特性。更多详情请查看[昇腾环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_009.html)
```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

#### 修改服务化参数
```
cd /usr/local/Ascend/mindie/latest/mindie-service/
vim conf/config.json
```
修改以下参数：
```
"httpsEnabled" : false, # 如果网络环境不安全，不开启HTTPS通信，即“httpsEnabled”=“false”时，会存在较高的网络安全风险
...
# 若不需要安全认证，则将以下两个参数设为false
"interCommTLSEnabled" : false,
"interNodeTLSEnabled" : false,
...
"npudeviceIds" : [[0,1,2,3,4,5,6,7]],
...
"modelName" : "qwen" # 不影响服务化拉起
"modelWeightPath" : "权重路径",
"worldSize":8,
```
Example：仅供参考，请根据实际情况修改。
```
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindie-server.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.1",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000, 
        "httpsEnabled" : false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : false,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm"
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 10000,
            "maxInputTokenLen" : 2048,
            "truncation" : true,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "qwen",
                    "modelWeightPath" : "/home/data/qwen2.5-7B-Instruct",
                    "worldSize" : 8,
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 8,
            "maxPrefillTokens" : 2048,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 8,
            "maxIterTimes" : 1024,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

#### 拉起服务化
```
# 以下命令需在所有机器上同时执行
# 解决权重加载过慢问题
export OMP_NUM_THREADS=1
# 设置显存比
export NPU_MEMORY_FRACTION=0.95
# 拉起服务化
cd /usr/local/Ascend/mindie/latest/mindie-service/
./bin/mindieservice_daemon
```
执行命令后，首先会打印本次启动所用的所有参数，然后直到出现以下输出：
```
Daemon start success!
```
则认为服务成功启动。


#### 另起客户端
进入相同容器，向服务端发送请求。

更多信息可参考官网信息：[MindIE Service](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0285.html)

### 精度化测试样例

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
export MINDIE_LOG_TO_STDOUT=1
chmod 640 /usr/local/lib/python3.11/site-packages/mindiebenchmark/config/config.json
```

需要开启确定性计算环境变量。
```
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
```
-并发数需设置为1，确保模型推理时是1batch输入，这样才可以和纯模型比对精度。
-使用MMLU比对精度时，MaxOutputLen应该设为20，MindIE Server的config.json文件中maxSeqLen需要设置为3600，该数据集中有约为1.4w条数据，推理耗时会比较长。
```
benchmark \
--DatasetPath "/数据集路径/MMLU" \
--DatasetType mmlu \
--ModelName qwen \
--ModelPath "/模型权重路径/Qwen2.5" \
--TestType client \
--Http http://{ipAddress}:{port} \
--ManagementHttp http://{managementIpAddress}:{managementPort} \
--Concurrency 1 \
--MaxOutputLen 20 \
--TaskKind stream \
--Tokenizer True \
--TestAccuracy True
```
ModelName，ModelPath需要与mindie-service里的config.json里的一致。样例仅供参考，请根据实际情况调整参数。

## 附录

#### 数据集下载

- 首先，需要在test/modeltest路径下新建名为temp_data的文件目录，然后在temp_data文件目录下新建对应数据集文件目录:

|    支持数据集  |     目录名称   |
|---------------|---------------|
|      BoolQ    |     boolq     |
|    HumanEval  |   humaneval   |
|   HumanEval_X |  humaneval_x  |
|      GSM8K    |     gsm8k     |
|   LongBench   |   longbench   |
|       MMLU    |     mmlu      |
|  NeedleBench  |   needlebench |
|  VideoBench   |   VideoBench  |
|  Vocalsound   |   Vocalsound  |
|   TextVQA     |   TextVQA     |
|   TruthfulQA  |   truthfulqa  |

- 获取数据集：需要访问huggingface和github的对应网址，手动下载对应数据集

|    支持数据集   |         下载地址            |
|----------------|-----------------------------|
|   BoolQ   |[dev.jsonl](https://storage.cloud.google.com/boolq/dev.jsonl)|
| HumanEval |[humaneval](https://github.com/openai/human-eval/raw/refs/heads/master/data/HumanEval.jsonl.gz)|
|HumanEval_X|[cpp](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/cpp/data)<br>[java](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/java/data)<br>[go](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/go/data)<br>[js](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/js/data)<br>[python](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/python/data)|
|  GSM8K    |[gsm8k](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl)|
| LongBench |[longbench](https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip)|
|    MMLU   |[mmlu](https://people.eecs.berkeley.edu/~hendrycks/data.tar)|
|NeedleBench|[PaulGrahamEssays](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[multi_needle_reasoning_en](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[multi_needle_reasoning_zh](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[names](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[needles](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_finance](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_game](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_general](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_government](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_movie](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_tech](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)|
|TextVQA|[train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)<br>[textvqa_val.jsonl](https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val.jsonl)<br>[textvqa_val_annotations.json](https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json)<br>|
|VideoBench|[Eval_QA/](https://github.com/PKU-YuanGroup/Video-Bench)<br>[Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench/tree/main)<br>|
|VocalSound|[VocalSound 16kHz Version](https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1)<br>|
|TruthfulQA|[truthfulqa](https://huggingface.co/datasets/domenicrosati/TruthfulQA/tree/main)|

- 将对应下载的数据集文件放置在对应的数据集目录下，并在modeltest根目录`MindIE-LLM/examples/atb_models/tests/modeltest`下执行：

```bash
python3 scripts/data_prepare.py [可选参数]
```

| 参数名  | 含义                     |
|--------|------------------------------|
| dataset_name | 可选，需要下载的数据集名称，支持的数据集列表参见[**功能**]章节，多个名称以','隔开                 |
| remove_cache | 可选，是否在下载前清除数据集缓存    |

## FAQ

- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`
- Qwen2.5系列模型当前800I A2采用bf16， 300I DUO使用fp16 

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。