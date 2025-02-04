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

# Qwen2-VL-7B-Instruct
Qwen2-VL-7B-Instruct 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM），可以以图像、文本、视频作为输入，并以文本作为输出
## 准备模型
目前提供的MindIE镜像预置了 Qwen2-VL-7B-Instruct 模型推理脚本，无需使用本仓库自带的atb_models中的代码
## 加载镜像
前往[昇腾社区/开发资源](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)下载适配本模型的镜像包：1.0.0-800I-A2-py311-openeuler24.03-lts

完成之后，请使用`docker images`命令确认查找具体镜像名称与标签。 

## 硬件要求
部署Qwen2-VL-7B-Instruct模型至少需要1台800I A2 32G服务器

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
pip install -r requirements/models/requirements_qwen2_vl.txt
```

# 纯模型推理


- 修改`/usr/local/Ascend/atb-models/examples/models/qwen2_vl/run_pa.sh`脚本

```shell
# 设置卡数，800I-A2-32G必须八卡，800I-A2-64G四卡八卡均可
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

...

# 模型权重路径
model_path="/data/Qwen2-VL-7B-Instruct/"
# 批次大小，底层使用continuous batching逻辑
max_batch_size=1
# 最大输入长度，输入长视频或者较大分辨率图片时，需要设置较大的值，以便支持更长的输入序列
# kv cache会根据，最大输入长度、最大输出长度以及bs进行预分配，设置太大会影响吞吐
max_input_length=8192
# 最大输出长度
max_output_length=80
# 单张图或单个图片
input_image="XXX.jpg/png/jpeg/mp4/wmv/avi" 
# 用户prompt，默认放置在图片后
input_text="Explain the details in the image."
# dataset_path优先级比input_image高，若要推理整个数据集，base_cmd入参中添加 ```--dataset_path $dataset_path \```
dataset_path="/data/test_images" 
# 共享内存name保存路径，任意位置的一个txt即可
shm_name_save_path="./shm_name.txt"

```

- 运行脚本

```shell
bash /usr/local/Ascend/atb-models/examples/models/qwen2_vl/run_pa.sh
```

- 性能测试样例（800I A2 32G）
  
  - 设置`max_batch_size=4`
  - 设置`max_input_length=8192`
  - 设置`max_output_length=80`
  - 设置`input_image="/XXX/1902x1080.jpg"`
  - 运行`run_pa.sh`脚本
  - 输出结果为，吞吐即为 320 / 7.44 = 43 tokens/s
    ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202412305586000/16410584/6d45582d06814674a1b0190af4dfa9f6.png)
  - 更详细的性能数据，如首token时延，参考终端performance输出
    ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202412305586000/16413152/97db1bd60cb74f5cafc532b27684d1c0.png)
- 性能测试样例（800I A2 64G）
  
  - 设置`max_batch_size=32`
  - 设置`max_input_length=8192`
  - 设置`max_output_length=80`
  - 设置`input_image="/XXX/1902x1080.jpg"`
  - 运行`run_pa.sh`脚本
  - 输出结果为，吞吐即为 2560 / 25.912 = 98.79 tokens/s
    ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202412305586000/16423637/5ec9803cb50b4e869c0463709c8d09e1.png)
  - 更详细的性能数据，如首token时延，参考终端performance输出
    ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202412305586000/16423685/b9e88e350d0c42cd9346bf38cc3fc916.png)

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
"modelName" : "qwen2_vl", # 为了方便使用benchmark测试，modelname建议使用qwen2_vl
"modelWeightPath" : "/data/datasets/Qwen2-VL-7B-Instruct",
"worldSize" : 8,
...
"npuMemSize" : 8, #kvcache分配，可自行调整，单位是GB，切勿设置为-1，需要给vit预留显存空间
...
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
"model": "qwen2_vl"
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