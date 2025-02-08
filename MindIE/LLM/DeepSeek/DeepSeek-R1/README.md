
# DeepSeek-R1

## 权重

**权重下载**

- [Deepseek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main)
- [Deepseek-R1-Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero/tree/main)


**权重转换（Convert FP8 weights to BF16）**
1. GPU侧权重转换
```sh
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd DeepSeek-V3/inferece/
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/DeepSeek-R1 --output-bf16-hf-path /path/to/deepseek-R1-bf16 
```
注意：DeepSeek官方没有针对DeepSeek-R1提供新的权重转换脚本，所以复用DeepSeek-V3的权重转换脚本

2. NPU侧权重转换
目前npu转换脚本不会自动复制tokenizer等文件
```sh
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch\MindIE\LLM\DeepSeek\DeepSeek-V2\NPU_inference
python fp8_cast_bf16.py --input-fp8-hf-path /path/to/DeepSeek-R1 --output-bf16-hf-path /path/to/deepseek-R1-bf16
```

注意：
- `/path/to/DeepSeek-R1` 表示DeepSeek-R1原始权重路径，`/path/to/deepseek-R1-bf16` 表示权重转换后的新权重路径
- 由于模型权重较大，请确保您的磁盘有足够的空间放下所有权重，例如DeepSeek-R1在转换前权重约为640G左右，在转换后权重约为1.3T左右
- 推理作业时，也请确保您的设备有足够的空间加载模型权重，并为推理计算预留空间

**量化权重生成**

详情请参考 [DeepSeek模型量化方法介绍](https://gitee.com/ascend/msit/tree/br_noncom_MindStudio_8.0.0_POC_20251231/msmodelslim/example/DeepSeek)

目前支持：
- 生成模型w8a16量化权重，使用histogram量化方式，在CPU上进行运算
- 生成模型w8a8混合量化权重，使用histogram量化方式 (MLA:w8a8量化，MOE:w8a8 dynamic pertoken量化)

注意：DeepSeek-R1模型权重较大，量化权重生成时间较久，请耐心等待；具体时间与校准数据集大小成正比，10条数据大概需花费3小时。

### 加载镜像
前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配

DeepSeek-R1的镜像包：mindie_2.0.T3-800I-A2-py311-openeuler24.03-lts-aarch64.tar.gz
镜像加载后的名称：mindie:2.0.T3-800I-A2-py311-openeuler24.03-lts-aarch64

注意：量化需要使用mindie:2.0.T3版本

完成之后，请使用`docker images`命令确认查找具体镜像名称与标签。 
```
docker load -i mindie:2.0.T3-800I-A2-py311-openeuler24.03-lts-aarch64(下载的镜像名称与标签)
```

各组件版本配套如下：
| 组件 | 版本 |
| - | - |
| MindIE | 2.0.T3 |
| CANN | 8.0.T63 |
| PTA | 6.0.T700 |
| MindStudio | Msit: br_noncom_MindStudio_8.0.0_POC_20251231分支 |
| HDK | 24.1.0 |

## 硬件要求
部署DeepSeek-R1模型用BF16权重进行推理至少需要4台Atlas 800I A2（8\*64G）服务器，用W8A8量化权重进行推理则至少需要2台Atlas 800I A2 (8\*64G)

### 容器启动
#### 1. 准备模型
目前提供的MindIE镜像预置了DeepSeek-R1模型推理脚本，无需再下载模型代码，也无需参考目录结构。（可跳过至获取模型权重）

- 下载对应模型代码，可以使用：
```sh
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```


目录结构应为如下：
```sh
├── DeepSeek-R1
│   ├── README.md
```
- 获取模型权重
   - 本地已有模型权重
      从您信任的来源自行获取权重后，放置在从上述下载的模型代码的主目录下，放置后的目录结构应为如下：
      ```sh
      ├── DeepSeek-R1
      │   ├── README.md
      │   └── 权重文件1
      │   .   
      │   .
      │   └── 权重文件n
      ```
   - 本地没有模型权重
      我们提供模型权重下载脚本，支持HuggingFace，ModelScope以及Modelers来源的模型下载，用法如下

      注意：以下引用的`atb_models`路径在`DeepSeek-V2`路径下

      1. 确认`atb_models/build/weights_url.yaml`文件中对应repo_id，当前已默认配置模型官方认可的下载地址，如您有其他信任来源的repo_id，可自行修改，默认配置如下：

      ```sh
      HuggingFace: deepseek-ai/DeepSeek-R1
      ModelScope: deepseek-ai/DeepSeek-R1
      Modelers: None
      ```
      2. 执行下载脚本`atb_models/build/download_weights.py`:
      
      | 参数名  | 含义                                             |
      |--------|--------------------------------------------------|
      | hub | 可选，str类型参数，hub来源，支持HuggingFace, ModelScope, Modelers  |
      | repo_id | 可选，str类型参数，仓库ID，默认从weight_url.yaml中读取    |
      | target_dir | 可选，str类型参数，默认放置在atb_models同级目录下            |


- 修改模型文件夹属组为1001，执行权限为750，执行：
```sh
chown -R 1001:1001 /path-to-weights/DeepSeek-R1
chmod -R 750 /path-to-weights/DeepSeek-R1
```
#### 2. 启动容器

- 执行以下启动命令（参考）：
```sh
docker run -itd --privileged  --name=容器名称 --net=host \
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
   -v /权重路径:/权重路径 \
   mindie:1.0.0-XXX-800I-A2-arm64-py3.11（根据加载的镜像名称修改） \
   bash
```
#### 开启通信环境变量
```
export ATB_LLM_HCCL_ENABLE=1
export ATB_LLM_COMM_BACKEND="hccl"
export HCCL_CONNECT_TIMEOUT=7200
export WORLD_SIZE=32
export HCCL_EXEC_TIMEOUT=0
```

### 纯模型测试
#### 前置准备
- 修改权重目录下config.json文件
```
将 model_type 更改为 deepseekv2 (全小写且无空格)
```
注意：在本仓实现中，DeepSeek-R1目前沿用DeepSeek-V2代码框架
- 检查机器网络情况
```
# 检查物理链接
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
# 检查链接情况
for i in {0..7}; do hccn_tool -i $i -link -g ; done
# 检查网络健康情况
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
# 查看侦测ip的配置是否正确
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
# 查看网关是否配置正确
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
# 检查NPU底层tls校验行为一致性，建议全0
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
# NPU底层tls校验行为置0操作
for i in {0..7};do hccn_tool -i $i -tls -s enable 0;done
```
- 获取每张卡的ip地址
```
for i in {0..7};do hccn_tool -i $i -ip -g; done
```
- 参考如下格式，配置rank_table_file.json
```
{
   "server_count": "...", # 总节点数
   # server_list中第一个server为主节点
   "server_list": [
      {
         "device": [
            {
               "device_id": "...", # 当前卡的本机编号，取值范围[0, 本机卡数)
               "device_ip": "...", # 当前卡的ip地址，可通过hccn_tool命令获取
               "rank_id": "..." # 当前卡的全局编号，取值范围[0, 总卡数)
            },
            ...
         ],
         "server_id": "...", # 当前节点的ip地址
         "container_ip": "..." # 容器ip地址（服务化部署时需要），若无特殊配置，则与server_id相同
      },
      ...
   ],
   "status": "completed",
   "version": "1.0"
}
```

#### 精度测试
- 进入modeltest路径
```
cd /usr/local/Ascend/llm_model/tests/modeltest/
```
- 运行测试脚本
```
# 需在所有机器上同时执行
bash run.sh pa_bf16 [dataset] ([shots]) [batch_size] [model_name] ([is_chat_model]) [weight_dir] [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address]
```
Example: 在DeepSeek-R1跑CEVAl数据集主节点的命令
```
bash run.sh pa_bf16 full_CEval 5 16 deepseekv2 /path/to/weights/DeepSeek-R1 /path/to/xxx/ranktable.json 32 4 0 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```
参数说明：
1. `dataset`可选full_BoolQ、full_CEval等，部分数据集需要设置`shots`
2. `model_name`为`deepseekv2`
3. `weight_dir`为模型权重路径
4. `rank_table_file`为“前置准备”中配置的`rank_table_file.json`路径
5. `world_size`为总卡数
6. `node_num`为当前节点编号，即`rank_table_file.json`的`server_list`中顺序确定
7. `rank_id_start`为当前节点起始卡号，即`rank_table_file.json`中当前节点第一张卡的`rank_id`
8. `master_address`为主节点ip地址，即`rank_table_file.json`的`server_list`中第一个节点的ip

#### 性能测试
- 进入modeltest路径
```
cd /usr/local/Ascend/llm_model/tests/modeltest/
```
- 运行测试脚本
```
# 需在所有机器上同时执行
bash run.sh pa_bf16 performance [case_pair] [batch_size] [model_name] ([is_chat_model]) [weight_dir] [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address]
```
参数含义同“精度测试”

Example: 在DeepSeek-R1跑性能测试主节点的命令
```
bash run.sh pa_bf16 performance [[256,256]] 16 deepseekv2 /path/to/weights/DeepSeek-R1 /path/to/xxx/ranktable.json 32 4 0 {主节点IP}
# 0 代表从0号卡开始推理，之后的机器依次从8，16，24。
```

### 服务化测试
#### 配置服务化环境变量

变量含义：expandable_segments-使能内存池扩展段功能，即虚拟内存特性。更多详情请查看[昇腾环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_009.html)
```
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
服务化需要`rank_table_file.json`中配置`container_ip`字段
所有机器的配置应该保持一致，除了环境变量的MIES_CONTAINER_IP为本机ip地址。
```
export MIES_CONTAINER_IP=容器ip地址
export RANKTABLEFILE=rank_table_file.json路径
```

#### 修改服务化参数
```
cd /usr/local/Ascend/mindie/latest/mindie-service/
vim conf/config.json
```
修改以下参数
```
"httpsEnabled" : false,
...
"multiNodesInferEnabled" : true, # 开启多机推理
...
# 若不需要安全认证，则将以下两个参数设为false
"interCommTLSEnabled" : false,
"interNodeTLSEnabled" : false,
...
"modelName" : "DeepSeek-R1" # 不影响服务化拉起
"modelWeightPath" : "权重路径",
```
Example：仅供参考，不保证性能
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
        "ipAddress" : "改成主节点IP",
        "managementIpAddress" : "改成主节点IP",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000, //如果是4机，建议300
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
            "maxSeqLen" : 10000,
            "maxInputTokenLen" : 2048,
            "truncation" : true,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "deepseekr1",
                    "modelWeightPath" : "/home/data/dsR1_base_step178000",
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



#### 来到客户端
进入相同容器，向服务端发送请求。

更多信息可参考官网信息：[MindIE Service](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindieservice/servicedev/mindie_service0001.html)


### 常见问题

#### 服务化常见问题
1. 若出现out of memory报错，可适当调高NPU_MEMORY_FRACTION环境变量（默认值为0.8），适当调低服务化配置文件config.json中maxSeqLen、maxInputTokenLen、maxPrefillBatchSize、maxPrefillTokens、maxBatchSize等参数
```
export NPU_MEMORY_FRACTION=0.96
```
2. 若出现hccl通信超时报错，可配置以下环境变量
```
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=0
```
3. 若出现AttributeError：'IbisTokenizer' object has no atrribute 'cache_path'
Step1: 进入环境终端后执行
```
pip show mies_tokenizer
```
默认出现类似如下结果，重点查看`Location`
```
Name: mies_tokenizer
Version: 0.0.1
Summary: ibis tokenizer
Home-page:
Author:
Author-email:
License:
Location: /usr/local/python3.10.13/lib/python3.10/site-packages
Requires:
Required-by:
```
Step2: 打开`Location`路径下的./mies_tokenizer/tokenizer.py文件
```
vim /usr/local/python3.10.13/lib/python3.10/site-packages/mies_tokenizer/tokenizer.py
```
Step3: 对以下两个函数代码进行修改
```
def __del__(self):
-       dir_path = file_utils.standardize_path(self.cache_path)
+       cache_path = getattr(self, 'cache_path', None)
+       if cache_path is None:
+           return
+       dir_path = file_utils.standardize_path(cache_path)
        file_utils.check_path_permission(dir_path)
        all_request = os.listdir(dir_path)
```
以及
```
def _get_cache_base_path(child_dir_name):
        dir_path = os.getenv("LOCAL_CACHE_DIR", None)
        if dir_path is None:
           dir_path = os.path.expanduser("~/mindie/cache")
-          if not os.path.exists(dir_path):
-              os.makedirs(dir_path)
+          os.makedirs(dir_path, exist_ok=True)
           os.chmod(dir_path, 0o750)
```
4. 若出现`UnicodeEncodeError: 'ascii' codec can't encode character `\uff5c` in position 301:ordinal not in range(128)`

这是因为由于系统在写入或打印日志ASCII编码deepseek的词表失败，导致报错，不影响服务化正常运行。如果需要规避，需要/usr/local/Ascend/atb-models/atb_llm/runner/model_runner.py的第145行注释掉：print_log(rank, logger.info, f'init tokenizer done: {self.tokenizer}')

#### 权重路径权限问题
注意保证权重路径是可用的，执行以下命令修改权限，**注意是整个父级目录的权限**：
```sh
chown -R HwHiAiUser:HwHiAiUser /path-to-weights
chmod -R 750 /path-to-weights
```
