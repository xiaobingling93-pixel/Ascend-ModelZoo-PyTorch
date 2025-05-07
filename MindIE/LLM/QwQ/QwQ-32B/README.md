# 通义千问 QwQ-32B
## 简介
QwQ 是 Qwen 系列的推理模型。与传统的指令调优模型相比，具备思考和推理能力的 QwQ 在下游任务中，特别是在解决难题时，能够显著提高性能。QwQ-32B 是一个中等规模的推理模型，其性能可以与当前先进的推理模型（例如 DeepSeek-R1、o1-mini）相媲美。

## 权重

**权重下载**

- [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B)


## 加载镜像
前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts

完成加载镜像后，请使用`docker images`命令确认查找具体镜像名称与标签。
```shell
docker load -i mindie:1.0.0-800I-A2-py311-openeuler24.03-lts(下载的镜像名称与标签)
```

镜像中各组件版本配套如下：
| 组件 | 版本 |
| - | - |
| MindIE | 1.0.0 |
| CANN | 8.0.0 |
| PTA | 6.0.0 |
| MindStudio | 7.0.0 |
| HDK | 24.1.0 |

## 约束条件
- 部署QwQ-32B模型至少需要1台Atlas 800I A2 32G服务器
- 当前支持TP=4/8推理
- /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json 的权限默认为640, 请不要更改该文件权限。

## 新建容器

目前提供的MindIE镜像预置了QwQ-32B模型推理脚本，无需再额外下载魔乐仓库承载的模型适配代码，直接新建容器即可。

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
    mindie:1.0.0-800I-A2-py311-openeuler24.03-lts bash
```

如果您希望使用自行构建的普通用户镜像，并且规避容器相关权限风险，可以使用以下命令指定用户与设备：
```sh
docker run -it -d --net=host --shm-size=1g \
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
    mindie:1.0.0-800I-A2-py311-openeuler24.03-lts bash
```
更多镜像使用信息请参考[官方镜像仓库文档](https://gitee.com/ascend/ascend-docker-image/tree/dev/mindie#%E5%90%AF%E5%8A%A8%E5%AE%B9%E5%99%A8)。

## 进入容器
```shell
docker exec -it ${容器名称} bash
```

## 权重量化
### Atlas 800I A2 w8a8量化
W8A8量化权重可通过[msmodelslim](https://gitee.com/ascend/msit/blob/master/msmodelslim/example/Qwen/README.md)（昇腾压缩加速工具）实现。
- 注意该量化方式仅支持在Atlas 800I A2服务器上运行
- 环境配置请参考[使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)
- git clone下载msit仓代码； `git clone https://gitee.com/ascend/msit.git`
- 进入到msit/msmodelslim的目录 `cd msit/msmodelslim`；并在进入的msmodelslim目录下，运行安装脚本 `bash install.sh`;
- 进入到msit/msmodelslim/example/Qwen的目录 `cd msit/msmodelslim/example/Qwen`；并在进入的Qwen目录下，运行量化转换脚本
```bash
python3 quant_qwen.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m1
```
- 请将{浮点权重路径}和{量化权重路径}替换为用户实际路径。
- 如果需要使用npu多卡量化，请先配置环境变量，支持多卡量化,建议四卡执行量化：
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
```

## 纯模型推理

### 依赖配置
transformers版本升级至4.45.0，或将tokenizers版本升级至0.20.0。

### 对话测试
进入llm_model路径

ATB_SPEED_HOME_PATH默认/usr/local/Ascend/llm_model,以情况而定

```shell
cd $ATB_SPEED_HOME_PATH
```

执行对话测试

```shell
torchrun --nproc_per_node 4 \
         --master_port 20037 \
         -m examples.run_pa \
         --model_path {权重路径} \
         --trust_remote_code
         --max_output_length 32
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
具体执行batch=1, 输入长度256, 输出长度256用例的4卡并行性能测试命令为：
```shell
bash run.sh pa_bf16 performance [[256,256]] 1 qwen ${weight_path} 4
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
"npuDeviceIds" : [[0,1,2,3]],
...
"ModelDeployConfig":
{
"truncation" : false,
"ModelConfig" : [
{
...
"modelName" : "qwen2",
"modelWeightPath" : "/data/datasets/QwQ-32B",
"worldSize" : 4,
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
"max_tokens": 32,
"stream": false,
"do_sample": true,
"repetition_penalty": 1.00,
"temperature": 0.6,
"top_p": 0.6,
"top_k": 20,
"model": "qwen2"
}'
```

> 注: 服务化推理的更多信息请参考[MindIE Service用户指南](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0001.html)

## 常见问题
1. ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'. 降低transformers版本可解决。

```shell
pip install transformers==4.46.3 --force-reinstall
pip install numpy==1.26.4 --force-reinstall
```
2. Exception: data did not match any variant of untagged enum ModelWrapper at line 757479 column 3. 升级tokenizers版本可解决。
```shell
pip install tokenizers==0.20.0 --force-reinstall
```

## 声明
- 本代码仓提到的数据集和模型仅作为示例,这些数据集和模型仅供您用于非商业目的,如您使用这些数据集和模型来完成示例,请您特别注意应遵守对应数据集和模型的License,如您因使用数据集或模型而产生侵权纠纷,华为不承担任何责任。
- 如您在使用本代码仓的过程中,发现任何问题(包括但不限于功能问题、合规问题),请在本代码仓提交issue,我们将及时审视并解答。