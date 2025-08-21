# Qwen3-235B-A22B

## 硬件要求
部署Qwen3-235B-A22B模型进行推理至少需要2台Atlas 800I A2（8\*64G）服务器

## 权重
**权重下载**
### BF16原始权重下载
- [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B)
- [HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B)

## 推理前置准备

- 检查机器网络情况
```
# 1.检查物理链接
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
```
```
# 2.检查链接情况
for i in {0..7}; do hccn_tool -i $i -link -g ; done
```
```
# 3.检查网络健康情况
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
```
```
# 4.查看侦测ip的配置是否正确
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
```
```
# 5.查看网关是否配置正确
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
```
```
# 6.检查NPU底层tls校验行为一致性，建议统一全部设置为0，避免hccl报错
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
```
```
# 7.NPU底层tls校验行为置0操作，建议统一全部设置为0，避免hccl报错
for i in {0..7};do hccn_tool -i $i -tls -s enable 0;done
```
- 获取每张卡的ip地址
```
for i in {0..7};do hccn_tool -i $i -ip -g; done
```
- 需要用户自行创建rank_table_file.json，参考如下格式配置

以下是一个双机用例，用户自行添加ip，补全device：
```
{
   "server_count": "2",
   "server_list": [
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "...",
               "rank_id": "0"
            },
            {
               "device_id": "1",
               "device_ip": "...",
               "rank_id": "1"
            },
            ...
            {
               "device_id": "7",
               "device_ip": "...",
               "rank_id": "7"
            },
         ],
         "server_id": "...",
         "container_ip": "..."
      },
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "...",
               "rank_id": "8"
            },
            {
               "device_id": "1",
               "device_ip": "...",
               "rank_id": "9"
            },
            ...
            {
               "device_id": "7",
               "device_ip": "...",
               "rank_id": "15"
            },
         ],
         "server_id": "...",
         "container_ip": "..."
      },
   ],
   "status": "completed",
   "version": "1.0"
}
```
| 参数          |  说明                                                       |
|---------------|------------------------------------------------------------|
|  server_count |  总节点数                                                   |
|  server_list  |  server_list中第一个server为主节点                           |
|  device_id    |  当前卡的本机编号，取值范围[0, 本机卡数)                       |
|  device_ip    |  当前卡的ip地址，可通过hccn_tool命令获取                       |
|  rank_id      |  当前卡的全局编号，取值范围[0, 总卡数)                         |
|  server_id    |  当前节点的ip地址                                             |
|  container_ip |  容器ip地址（服务化部署时需要），若无特殊配置，则与server_id相同 |

rank_table_file.json配置完成后，需要执行命令修改权限为640
```sh
chmod -R 640 {rank_table_file.json路径}
```

- 修改模型文件夹属组为1001 -HwHiAiUser属组（容器为Root权限可忽视），执行权限为750：
```sh
chown -R 1001:1001 {/path-to-weights/Qwen3-235B-A22B}
chmod -R 750 {/path-to-weights/Qwen3-235B-A22B}
```

## 加载镜像

前往[昇腾社区/开发资源](https://support.huawei.com/enterprise/zh/ascend-computing/mindie-pid-261803968/software/266130647?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|261803968)下载适配本模型的镜像包mindie:2.2.T10-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz

```shell
docker load -i mindie_2.0.T18.B010-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz(下载的镜像名称与标签)
```

完成之后，请使用`docker images`命令确认查找具体镜像名称与标签。
```
docker images
```

## 容器启动

#### 启动容器

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
   -v  {/rank_table_file.json路径:/rank_table_file.json路径}  \
    {mindie:2.2.T10-800I-A2-py3.11-openeuler24.03-lts-aarch64（根据加载的镜像名称修改）}  \
   bash
```
#### 进入容器

- 执行以下命令进入容器（参考）：
```sh
docker exec -it {容器名称} bash
```
-  进入容器后需升级`transformers`版本
```sh
pip install transformers==4.51.0
```

#### 设置基础环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
```
#### 开启通信环境变量
```
export ATB_LLM_HCCL_ENABLE=1
export ATB_LLM_COMM_BACKEND="hccl"
export HCCL_CONNECT_TIMEOUT=7200 # 该环境变量需要配置为整数，取值范围[120,7200]，单位s
export HCCL_EXEC_TIMEOUT=0
```

## 纯模型推理
【使用场景】使用相同输入长度和相同输出长度，构造多Batch去测试纯模型性能

#### 精度测试
- 进入modeltest路径
```
cd /usr/local/Ascend/atb-models/tests/modeltest/
```
- 运行测试脚本

Step1.主副节点分别先清理残余进程：
```
pkill -9 -f 'mindie|python'
```
Step2.需在所有机器上同时执行：
```
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_model]) [weight_dir] [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address]
```
参数说明：
1. `data_type`：为数据类型，根据权重目录下config.json的data_type选择bf16或者fp16，例如：pa_bf16。
2. `dataset`：可选full_BoolQ、full_CEval等，相关数据集可至[魔乐社区MindIE](https://modelers.cn/MindIE)下载，（下载之前，需要申请加入组织，下载之后拷贝到/usr/local/Ascend/atb-models/tests/modeltest/路径下）CEval与MMLU等数据集需要设置`shots`（通常设为5）。
3. `batch_size`：为`batch数`。
4. `model_name`：为`qwen`。
5. `is_chat_model`：为`是否支持对话模式，若传入此参数，则进入对话模式`。
6. `weight_dir`：为模型权重路径。
7. `rank_table_file`：为“前置准备”中配置的`rank_table_file.json`路径。
8. `world_size`：为总卡数。
9. `node_num`：为当前节点编号，即`rank_table_file.json`的`server_list`中顺序确定。
10. `rank_id_start`：为当前节点起始卡号，即`rank_table_file.json`中当前节点第一张卡的`rank_id`，Atlas 800I-A2双机场景下，主节点为0，副节点为8。
11. `master_address`：为主节点ip地址，即`rank_table_file.json`的`server_list`中第一个节点的ip。

测试脚本运行如下，以双机为例：

样例 -CEval 带shot

主节点
```
bash run.sh pa_bf16 full_CEval 5 1 qwen {/path/to/weights/Qwen3-235B-A22B} {/path/to/xxx/ranktable.json} 16 2 0 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点
```
bash run.sh pa_bf16 full_CEval 5 1 qwen {/path/to/weights/Qwen3-235B-A22B} {/path/to/xxx/ranktable.json} 16 2 8 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```

样例 -GSM8K 不带shot

主节点
```
bash run.sh pa_bf16 full_GSM8K 8 qwen {/path/to/weights/Qwen3-235B-A22B} {/path/to/xxx/ranktable.json} 16 2 0 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点
```
bash run.sh pa_bf16 full_GSM8K 8 qwen {/path/to/weights/Qwen3-235B-A22B} {/path/to/xxx/ranktable.json} 16 2 8 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```

#### 性能测试
- 进入modeltest路径：
```
cd /usr/local/Ascend/atb-models/tests/modeltest/
```
Step1.主副节点分别先清理残余进程：
```
pkill -9 -f 'mindie|python'
```
Step2.需在所有机器上同时执行：
```
bash run.sh pa_[data_type] performance [case_pair] [batch_size] ([prefill_batch_size]) [model_name] ([is_chat_model]) [weight_dir] [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address]
```
参数说明：
1. `data_type`：为数据类型，根据权重目录下config.json的data_type选择bf16或者fp16，例如：pa_bf16。
2. `case_pair`：[最大输入长度,最大输出长度]。
3. `batch_size`：为`batch数`。
4. `prefill_batch_size`：为可选参数，设置后会固定prefill的batch size。
5. `model_name`：为`qwen`。
6. `is_chat_model`：为`是否支持对话模式，若传入此参数，则进入对话模式`。
7. `weight_dir`：为模型权重路径。
8. `rank_table_file`：为“前置准备”中配置的`rank_table_file.json`路径。
9. `world_size`：为总卡数。
10. `node_num`：为当前节点编号，即`rank_table_file.json`的`server_list`中顺序确定。
11. `rank_id_start`：为当前节点起始卡号，即`rank_table_file.json`中当前节点第一张卡的`rank_id`，Atlas 800I-A2双机场景下，主节点为0，副节点为8。
12. `master_address`：为主节点ip地址，即`rank_table_file.json`的`server_list`中第一个节点的ip。

测试脚本运行如下，以双机为例：

主节点
```
bash run.sh pa_bf16 performance [[256,256]] 1 qwen {/path/to/weights/Qwen3-235B-A22B} {/path/to/xxx/ranktable.json} 16 2 0 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
副节点
```
bash run.sh pa_bf16 performance [[256,256]] 1 qwen {/path/to/weights/Qwen3-235B-A22B} {/path/to/xxx/ranktable.json} 16 2 8 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```

## 服务化推理
【使用场景】对标真实客户上线场景，使用不同并发、不同发送频率、不同输入长度和输出长度分布，去测试服务化性能
#### 配置服务化环境变量

变量含义：expandable_segments-使能内存池扩展段功能，即虚拟内存特性。更多详情请查看[昇腾环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_009.html)。
```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
服务化需要`rank_table_file.json`中配置`container_ip`字段。
所有机器的配置应该保持一致，除了环境变量的MIES_CONTAINER_IP为本机ip地址。
```
export MIES_CONTAINER_IP={容器ip地址}
export RANKTABLEFILE={rank_table_file.json路径}
```

#### 修改服务化参数
```
cd /usr/local/Ascend/mindie/latest/mindie-service/
vim conf/config.json
```
修改以下参数
```
"httpsEnabled" : false, # 如果网络环境不安全，不开启HTTPS通信，即“httpsEnabled”=“false”时，会存在较高的网络安全风险
...
"multiNodesInferEnabled" : true, # 开启多机推理
...
# 若不需要安全认证，则将以下两个参数设为false
"interCommTLSEnabled" : false,
"interNodeTLSEnabled" : false,
...
"npudeviceIds" : [[0,1,2,3,4,5,6,7]],
...
"modelName" : "Qwen3-235B-A22B" # 不影响服务化拉起
"modelWeightPath" : "权重路径",
"worldSize":8,
```
Example：仅供参考，请根据实际情况修改
```
{
    "Version" : "1.0.0",

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
        "openAiSupport" : "vllm",
        "tokenTimeout" : 600,
        "e2eTimeout" : 600,
        "distDPServerEnabled":false
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : true,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : false,
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
            "maxSeqLen" : 2560,
            "maxInputTokenLen" : 2048,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "qwen-30b",
                    "modelWeightPath" : "/dl/weights/qwen/Qwen3-30B",
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

            "maxPrefillBatchSize" : 50,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
            "maxIterTimes" : 512,
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

更多信息可参考官网信息：[MindIE Service](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0285.html)。

### 精度化测试样例

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
--ModelName Qwen3-235B-A22B \
--ModelPath "/模型权重路径/Qwen3-235B-A22B" \
--TestType client \
--Http https://{ipAddress}:{port} \
--ManagementHttp https://{managementIpAddress}:{managementPort} \
--Concurrency 1 \
--MaxOutputLen 20 \
--TaskKind stream \
--Tokenizer True \
--TestAccuracy True
```
ModelName，ModelPath需要与mindie-service里的config.json里的一致，master_ip设置为主节点机器的ip。样例仅供参考，请根据实际情况调整参数。


### 常见问题

#### 服务化常见问题
1. 若出现out of memory报错，可适当调高NPU_MEMORY_FRACTION环境变量（默认值为0.8），适当调低服务化配置文件config.json中maxSeqLen、maxInputTokenLen、maxPrefillBatchSize、maxPrefillTokens、maxBatchSize等参数。
```
export NPU_MEMORY_FRACTION=0.96
```
2. 若出现hccl通信超时报错，可配置以下环境变量。
```
export HCCL_CONNECT_TIMEOUT=7200 # 该环境变量需要配置为整数，取值范围[120,7200]，单位s
export HCCL_EXEC_TIMEOUT=0
```

3. 从节点无法和主节点建立rpc通信

若出现多级部署从节点无法和主节点建立rpc通信问题，子节点报RPC问题，可能原因：防火墙拦截，排查方法：使用指令查看防火墙状态，如果开启防火墙，每台机器都需要关闭防火墙；

查看防火墙状态：
```
sudo systemctl status firewalld
```
临时关闭防火墙，该操作存在安全隐患，请谨慎操作，该命令适用于linux系统，其它系统需要根据实际情况修改：
```
sudo systemctl stop firewalld
```
参考链接：https://www.hiascend.com/document/caselibrary/detail/topic_0000002193154350

4. 无进程内存残留

如果卡上有内存残留，且有进程，可以尝试以下指令：
```
pkill -9 -f 'mindie|python'
```

如果卡上有内存残留，但无进程，可以尝试以下指令：
```
npu-smi set -t reset -i 0 -c 0 #重启npu卡
npu-smi info -t health -i <card_idx> -c 0 #查询npu告警
```

例：
```
npu-smi set -t reset -i 0 -c 0 #重启npu卡0
npu-smi info -t health -i 2 -c 0 #查询npu卡2告警
```
如果卡上有进程残留，无进程，且重启NPU卡无法消除残留内存，请尝试reboot重启机器

5. 日志收集

遇到推理报错时，请打开日志环境变量，收集日志信息。
- 算子库日志|默认输出路径为"~/atb/log"
```
export ASDOPS_LOG_LEVEL = INFO
export ASDOPS_LOG_TO_FILE = 1
```
- 加速库日志|默认输出路径为"~/mindie/log/debug"
```
export ATB_LOG_LEVEL = INFO
export ATB_LOG_TO_FILE = 1
```
- MindIE Service日志|默认输出路径为"~/mindie/log/debug"
```
export MINDIE_LOG_TO_FILE = 1
export MINDIE_LOG_TO_LEVEL = debug
```
- CANN日志收集|默认输出路径为"~/ascend"
```
export ASCEND_GLOBAL_LOG_TO_LEVEL = 1
```

6. 多机无法拉起Qwen3-MoE模型推理，HCCL报错
```
# 检查NPU底层tls校验行为一致性，建议统一全部设置为0，避免hccl报错
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
```
```
# NPU底层tls校验行为置0操作，建议统一全部设置为0，避免hccl报错
for i in {0..7};do hccn_tool -i $i -tls -s enable 0;done
```
#### 权重路径权限问题
注意保证权重路径是可用的，执行以下命令修改权限，**注意是整个父级目录的权限**：
```sh
chown -R HwHiAiUser:HwHiAiUser {/path-to-weights}
chmod -R 750 {/path-to-weights}
```
#### 更多故障案例，请参考链接：https://www.hiascend.com/document/caselibrary
## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集来完成示例，请您特别注意应遵守对应数据集合模型的License，如您因使用数据集或者模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本地代码的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。