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

# InternVL2_5-38B
InternVL2_5-38B 是一种多模态大模型，具有强大的图像和文本处理能力，通过开源组件缩小与商业多模态模型的差距——GPT-4V的开源替代方案。在聊天机器人中，InternVL可以通过解析用户的文字输入，结合图像信息，生成更加生动、准确的回复。 此外，InternVL还可以根据用户的图像输入，提供相关的文本信息，实现更加智能化的交互。

## 加载镜像
前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts

完成之后，请使用`docker images`命令确认查找具体镜像名称与标签。 

## 硬件要求
部署InternVL2_5-38B模型至少需要1台Atlas 800I A2 32G服务器

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

## 准备模型
首先需要更改dnf配置文件
```shell
vi /etc/dnf/dnf.conf 
```
在其中加一行: sslverify=false
```
[main]
gpgcheck=1
installonly_limit=3
clean_requirements_on_remove=True
best=True
skip_if_unavailable=False
sslverify=false # 新增
```
之后，用dnf工具安装git
```shell
dnf install git
```
最后，通过git clone命令拉取魔乐社区代码
```shell
git clone https://modelers.cn/MindIE/internvl2_5_38b.git
```
如果出现报错```server certificate verification failed. CAfile: none CRLfile: none```则需要增加一行
```shell
git config --global http.sslVerify false
```

目前提供的MindIE镜像预置了 InternVL2_5-38B 模型推理脚本，需将其中的部分代码加入至容器自带的atb-models中

```shell
# 清除原有代码
rm -rf /usr/local/Ascend/atb-models/atb_llm/models/internvl
# 将魔乐仓库自带的部分代码拷贝至至容器自带的atb-models中，前者为魔乐仓库的相对路径，可替换为绝对路径
cp -rf internvl2_5_38b/atb_models/atb_llm/models/internvl /usr/local/Ascend/atb-models/atb_llm/models/
```

## 安装Python依赖

```shell
cd /usr/local/Ascend/atb-models
pip install -r requirements/models/requirements_internvl.txt
```

# 纯模型推理


- 修改`/usr/local/Ascend/atb-models/examples/models/internvl/run_pa.sh`脚本

```shell
# 设置卡数，Atlas-800I-A2-32G必须八卡，Atlas-800I-A2-64G四卡八卡均可
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
"maxSeqLen" : 32768,
"maxInputTokenLen" : 32768,
"truncation" : false,
"ModelConfig" : [
{
"modelInstanceType": "Standard",
"modelName" : "internvl", # 为了方便使用benchmark测试，modelname建议使用internvl
"modelWeightPath" : "/data/datasets/InternVL2_5-38B",
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