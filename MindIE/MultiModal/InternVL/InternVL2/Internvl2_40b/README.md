---
language:
  - en
pipeline_tag: text-generation
tags:
  - pretrained
license: other
hardwares:
  - NPU
frameworks:
  - PyTorch
library_name: openmind
---

# InternVL2-40B
InternVL2-40B 是一种多模态大模型，具有强大的图像和文本处理能力，通过开源组件缩小与商业多模态模型的差距——GPT-4V的开源替代方案。在聊天机器人中，InternVL可以通过解析用户的文字输入，结合图像信息，生成更加生动、准确的回复。 此外，InternVL还可以根据用户的图像输入，提供相关的文本信息，实现更加智能化的交互。
## 准备模型
目前提供的MindIE镜像预置了 InternVL2-40B 模型推理脚本，无需使用本仓库自带的atb_models中的代码
## 加载镜像
前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts

完成之后，请使用`docker images`命令确认查找具体镜像名称与标签。 

## 硬件要求
部署InternVL2-40B模型至少需要1台Atlas 800I A2 推理服务器 32GB

## 新建容器

自行修改端口等参数，启动样例

```shell
docker run -dit -u root \
--name ${容器名} \
-e ASCEND_RUNTIME_OPTIONS=NODRV \
--privileged=true \
-v /home/路径:/home/路径 \
-v /data:/data \
-v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
-v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
-v /usr/local/sbin/:/usr/local/sbin \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--shm-size=100g \
-p ${映射端口}:22 \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
${MindIE 1.0.0 镜像} \
/bin/bash
```
## 进入容器
```shell
docker exec -it ${容器名} bash
```

## 安装Python依赖

```shell
cd /usr/local/Ascend/atb-models
pip install -r requirements/models/requirements_internvl.txt
```

# 纯模型推理


- 修改`/usr/local/Ascend/atb-models/examples/models/internvl/run_pa.sh`脚本

```shell
# 设置卡数，Atlas 800I A2 推理服务器 32GB上必须八卡，Atlas 800I A2 推理服务器 64GB上四卡八卡均可
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

- 运行脚本
可参考run_pa.sh同级目录下的README.md。
```shell
bash /usr/local/Ascend/atb-models/examples/models/internvl/run_pa.sh --run --trust_remote_code ${权重路径} ${图片或视频所在文件夹路径}
```

# 服务化推理


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
"npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
...
"ModelDeployConfig":
{
"maxSeqLen" : 50000,
"maxInputTokenLen" : 50000,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "internvl", # 为了方便使用benchmark测试，modelname建议使用internvl
"modelWeightPath" : "/data/datasets/InternVL2-40B",
"worldSize" : 8,
...
"npuMemSize" : 8, #kvcache分配，可自行调整，单位是GB，切勿设置为-1，需要给vit预留显存空间
...
"trustRemoteCode" : false #默认为false，若设为true，则信任本地代码，用户需自行承担风险
}
]
},
"ScheduleConfig" :
{
...
"maxPrefillTokens" : 50000,
"maxIterTimes": 4096,
...
}
}
}
```

- 设置运行多卡环境变量

```shell
export MASTER_ADDR=localhost 
export MASTER_PORT=7896
```

- 拉起服务化

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

- 容器内新端口测试 VLLM接口

```shell
curl 127.0.0.1:1040/generate -d '{
"prompt": [
{
"type": "image_url",
"image_url": ${图片路径}
},
{"type": "text", "text": "Explain the details in the image."}
],
"max_tokens": 512,
"stream": false,
"do_sample":true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1,
"model": "internvl"
}'
```

- 容器内新端口测试 OpenAI 接口

```shell
curl 127.0.0.1:1040/v1/chat/completions -d ' {
"model": "internvl",
"messages": [{
"role": "user",
"content": [
{"type": "image_url", "image_url": ${图片路径}},
{"type": "text", "text": "Explain the details in the image."}
]
}],
"max_tokens": 512,
"do_sample": true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1
}'
```
## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。