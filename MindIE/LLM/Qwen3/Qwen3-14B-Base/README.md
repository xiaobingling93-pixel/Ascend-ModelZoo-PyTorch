# Qwen3-14B-Base
## 简介
Qwen3是Qwen系列中最新一代的大型语言模型，提供了密集和混合专家(MoE)模型的全面套件。基于广泛的训练，Qwen3在推理、指令遵循、代理功能和多语言支持方面取得了很大的进展，主要具有以下功能：

- **思维模式**（用于复杂的逻辑推理、数学和编码）和**非思维模式**（用于高效、通用的对话）在单个模型内无缝切换，确保跨各种场景的最佳性能。
- **增强了推理能力**在数学、代码生成和常识逻辑推理方面超过了之前的QwQ（思维模式）和Qwen2.5（非思维模式）。
- **人类偏好调整**，擅长创意写作、角色扮演、多轮对话和指令跟随，提供更自然、更吸引人、更沉浸式的对话体验。
- **在代理能力方面的专业知识**，能够在思考模式和非思考模式下与外部工具精确集成，在基于代理的复杂任务中实现开源模型中的领先性能。
- **支持100多种语言和方言***具有强大多语言教学能力和翻译能力。

## 权重

**权重下载**

- [Qwen3-14B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-14B-Base/files)


## 加载镜像
前往[昇腾社区/开发资源](https://support.huawei.com/enterprise/zh/ascend-computing/mindie-pid-261803968/software/265528880?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|261803968)下载适配本模型的镜像包mindie:2.0.T10.B060-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz

```shelll
docker load -i mindie_2.0.T10.B060-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz(下载的镜像名称与标签)
```
完成加载镜像后，请使用`docker images`命令确认查找具体镜像名称与标签。

## 约束条件
- 当前支持TP=1/2/4/8推理
- /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json 的权限默认为640, 请不要更改该文件权限。

## 新建容器

目前提供的MindIE镜像预置了Qwen3-14B-Base模型推理脚本，无需再额外下载魔乐仓库承载的模型适配代码，直接新建容器即可。

执行以下启动命令（参考）：
如果您使用的是root用户镜像（例如从Ascend Hub上取得），并且可以使用特权容器，请使用以下命令启动容器：
```sh
docker run -it -d --net=host --shm-size=1g \
    --privileged \
    --name <container-name> \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path-to-weights:/path-to-weights:ro \
    mindie:2.0.T10.B060-800I-A2-py3.11-openeuler24.03-lts-aarch64 bash
```

如果您希望使用自行构建的普通用户镜像，并且规避容器相关权限风险，可以使用以下命令指定用户与设备：
```sh
docker run -it -d --net=host --shm-size=1g \
    --user mindieuser:<HDK-user-group> \
    --name <container-name> \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path-to-weights:/path-to-weights:ro \
    mindie:2.0.T10.B060-800I-A2-py3.11-openeuler24.03-lts-aarch64 bash
```
> 注意，以上启动命令仅供参考，请根据需求自行修改再启动容器，尤其需要注意：
>
> 1. `--user`，如果您的环境中HDK是通过普通用户安装（例如默认的`HwHiAiUser`，可以通过`id HwHiAiUser`命令查看该用户组ID），请设置好对应的用户组，例如用户组1001可以使用HDK，则`--user mindieuser:1001`，镜像中默认使用的是用户组1000。如果您的HDK是由root用户安装，且指定了`--install-for-all`参数，则无需指定`--user`参数。
>
> 2. 设定容器名称`--name`与镜像名称，例如800I A2服务器使用`mindie:2.0.T10.B060-800I-A2-py3.11-openeuler24.03-lts-aarch64`。
>
> 3. 设定想要使用的卡号`--device`。
>
> 4. 设定权重挂载的路径，`-v /path-to-weights:/path-to-weights:ro`，注意，如果使用普通用户镜像，权重路径所属应为镜像内默认的1000用户，且权限可设置为750。可使用以下命令进行修改：
>       ```sh
>       chown -R 1000:1000 /path-to-weights
>       chmod -R 750 /path-to-weights
>       ```
> 5. **在普通用户镜像中，注意所有文件均在 `/home/mindieuser` 下，请勿直接挂载 `/home` 目录，以免宿主机上存在相同目录，将容器内文件覆盖清除。**

## 进入容器
```shell
docker exec -it ${容器名称} bash
```

## 纯模型推理

### 依赖配置
transformers版本升级至4.51.0。

### 对话测试
进入atb-models路径, 并打开日志

ATB_SPEED_HOME_PATH默认/usr/local/Ascend/atb-models,以情况而定

```shell
cd $ATB_SPEED_HOME_PATH
export MINDIE_LOG_TO_STDOUT=1
```

执行对话测试

```shell
torchrun --nproc_per_node 2 \
         --master_port 20037 \
         -m examples.run_pa \
         --model_path {权重路径} \
         --trust_remote_code \
         --max_output_length 256
```

### 性能测试
进入ModelTest路径
```shell
cd $ATB_SPEED_HOME_PATH/tests/modeltest/
```
运行测试脚本
```shell
bash run.sh pa_[data_type] performance [case_pair] [batch_size] ([prefill_batch_size]) [model_name] ([is_chat_model]) (lora [lora_data_path]) [weight_dir] ([trust_remote_code]) [chip_num] ([parallel_params]) ([max_position_embedding/max_sequence_length])
```
具体执行batch=1, 输入长度256, 输出长度256用例的2卡并行性能测试命令为：
```shell
bash run.sh pa_bf16 performance [[256,256]] 1 qwen ${weight_path} 2
```

> 注：ModelTest为大模型的性能和精度提供测试功能。使用文档请参考`${ATB_SPEED_HOME_PATH}/tests/modeltest/README.md`


## 服务化推理


- 打开配置文件

```shell
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

- 更改配置文件

```json
{
...
"ServerConfig" :
{
...
"port" : 1040, #自定义
"managementPort" : 1041, #自定义
"metricsPort" : 1042, #自定义
...
"httpsEnabled" : false,
...
},

"BackendConfig": {
...
"npuDeviceIds" : [[0,1]],
...
"ModelDeployConfig":
{
"truncation" : false,
"ModelConfig" : [
{
...
"modelName" : "qwen3",
"modelWeightPath" : "/data/datasets/Qwen3-14B-Base",
"worldSize" : 2,
...
}
]
},
}
}
```

- 拉起服务化

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

- 新建窗口测试(OpenAI接口)

```shell
curl -X POST 127.0.0.1:1040/v1/chat/completions \
-d '{
"messages": [
{"role": "system", "content": "you are a helpful assistant."},
{"role": "user", "content": "How many r are in the word \"strawberry\""}
],
"max_tokens": 256,
"stream": false,
"do_sample": true,
"temperature": 0.6,
"top_p": 0.95,
"top_k": 20,
"model": "qwen3"
}'
```

> 注: 服务化推理的更多信息请参考[MindIE Service用户指南](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0001.html)


## 声明
- 本代码仓提到的数据集和模型仅作为示例,这些数据集和模型仅供您用于非商业目的,如您使用这些数据集和模型来完成示例,请您特别注意应遵守对应数据集和模型的License,如您因使用数据集或模型而产生侵权纠纷,华为不承担任何责任。
- 如您在使用本代码仓的过程中,发现任何问题(包括但不限于功能问题、合规问题),请在本代码仓提交issue,我们将及时审视并解答。