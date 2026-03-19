# Qwen3-Coder-30B-A3B
## 1.基本信息

| 项目| 信息 |
| --- | --- |
| 原始模型名 | Qwen3-Coder-30B-A3B-Instruct-w8a8 |
| 权重链接 | [权重链接](https://www.modelscope.cn/models/Eco-Tech/Qwen3-Coder-30B-A3B-Instruct-w8a8) |
| 测试机型 | Atlas 300I DUO |
| 版本 | MindIE |

OE镜像：[**quay.io/ascend/mindie:3.0.0b1-300I-Duo-py3.11-openeuler24.03-lts**](https://quay.io/repository/ascend/mindie?tab=tags)

## 2.启动容器

2.1启动容器命令

```
docker run -it -d --net=host --shm-size=1g
--name <container-name>
-w /home
--device=/dev/davinci0:rwm
--device=/dev/davinci1:rwm
--device=/dev/davinci2:rwm
--device=/dev/davinci3:rwm
--device=/dev/davinci_manager:rwm
--device=/dev/hisi_hdc:rwm
--device=/dev/devmm_svm:rwm
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
-v /usr/local/dcmi:/usr/local/dcmi:ro
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
-v /usr/local/sbin/:/usr/local/sbin:ro
-v /home/weight:/home/weight:ro
${IMAGES_ID} bash
```

2.2执行以下命令进入容器

```
docker exec -it <container-name> /bin/bash
```

## 3.模型推理

3.1执行如下命令，查询安装路径.

```
pip show mindie_llm | grep Location
```

若python版本是3.11，则查询到的默认安装路径为：/usr/local/lib/python3.11/site-packages。

3.2执行如下命令，进入安装路径。

```
cd <site-packages>
```

3.3确认配置文件有640权限。

```
chmod 640 <site-packages>/mindie_llm/conf/config.json
```

3.4设置环境变量，开启日志打印。

```
export MINDIE_LOG_TO_STDOUT=1
```

3.5配置服务化参数。
  进入conf目录，打开“config.json“文件。

```
cd mindie_llm/conf
vim config.json
```

根据实际情况修改“config.json“中的配置参数。（以下以Qwen3-Coder-30B-A3B为例）

```
{
    "ServerConfig" : 
        {
        "httpsEnabled" : false
        },
    "BackendConfig" : 
    {
            "npuDeviceIds" : [[0,1,2,3]],
            "ModelDeployConfig" :
        {
                "ModelConfig" : [
                {
                    "modelName" : "qwen3-coder",
                    "modelWeightPath" : "/home/weight",  
                    "worldSize" : 4,
                    "trustRemoteCode": false
                }
            ]
        },
    }
}

```
如上的参数说明如下，更多“config.json“的参数说明请参考[配置参数说明（服务化）。](https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/user_manual/service_parameter_configuration.md)

|配置项|取值类型|取值范围|配置说明|
|--|--|--|--|
|httpsEnabled|bool|true（开启）false（关闭）|是否开启HTTPS通信安全认证。true：开启HTTPS通信。false：关闭HTTPS通信。如果网络环境不安全，不开启HTTPS通信，即“httpsEnabled”=“false”时，会存在较高的网络安全风险。|
|npuDeviceIds|std::vector<std::set<size_t>>|根据模型和环境的实际情况来决定。|表示启用哪几张卡。对于每个模型实例分配的npuIds，使用芯片逻辑ID表示。在未配置ASCEND_RT_VISIBLE_DEVICES环境变量时，每张卡对应的逻辑ID可使用"npu-smi info -m"指令进行查询。若配置ASCEND_RT_VISIBLE_DEVICES环境变量时，可见芯片的逻辑ID按照ASCEND_RT_VISIBLE_DEVICES中配置的顺序从0开始计数。例如：ASCEND_RT_VISIBLE_DEVICES=1,2,3,4则以上可见芯片的逻辑ID按顺序依次为0,1,2,3。多机推理场景下该值无效，每个节点上使用的npuDeviceIds根据ranktable计算获得。必填，默认值：[[0,1,2,3]]。|
|modelName|string|由大写字母、小写字母、数字、中划线、点和下划线组成，且不以中划线、点和下划线作为开头和结尾，字符串长度小于或等于256。|模型名称。必填，默认值："llama_65b"。|
|modelWeightPath|std::string|文件绝对路径长度的上限与操作系统的设置（Linux为PATH_MAX）有关，最小值为1。|模型权重路径。程序会读取该路径下的config.json中torch_dtype和vocab_size字段的值，需保证路径和相关字段存在。必填，默认值："/data/atb_testdata/weights/llama1-65b-safetensors"。该路径会进行安全校验，需要和执行用户的属组和权限保持一致。|
|worldSize|uint32_t|根据模型实际情况来决定。每一套模型参数中worldSize必须与使用的NPU数量相等。|启用几张卡推理。必填，默认值：4。|
|trustRemoteCode|bool|truefalse|是否信任远程代码。false：不信任远程代码。true：信任远程代码。选填，默认值：false。如果设置为true，会存在信任远程代码行为，可能会导致恶意代码注入风险，请自行保障代码注入安全风险。|


3.6开启function call功能
  进入atb_llm/conf目录，打开“config.json“文件。

```
cd atb_llm/conf
vim config.json
```

修改tool_call_parser字段内容为qwen3_coder，如下所示

```
"tool_call_options": {
	"tool_call_parser": "qwen3_coder"
},
```


3.7修改模型配置文件
  进入模型权重目录，打开“config.json“文件。修改内容如下所示：

```
"max_position_embeddings": 40960,
"torch_dtype": "float16",
```

> 注：max_position_embeddings可根据内存限制自行增大或减小，最大不超默认值。

3.8启动服务

```
mindie_llm_server
```

回显如下则说明启动成功。

```
Daemon start success!
```

> 更多服务化配置说明请参考：[快速入门](https://gitcode.com/Ascend/MindIE-LLM/blob/master/docs/zh/user_guide/quick_start/quick_start.md#%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8)

## 声明
- 本代码仓提到的数据集和模型仅作为示例,这些数据集和模型仅供您用于非商业目的,如您使用这些数据集和模型来完成示例,请您特别注意应遵守对应数据集和模型的License,如您因使用数据集或模型而产生侵权纠纷,华为不承担任何责任。
- 如您在使用本代码仓的过程中,发现任何问题(包括但不限于功能问题、合规问题),请在本代码仓提交issue,我们将及时审视并解答。