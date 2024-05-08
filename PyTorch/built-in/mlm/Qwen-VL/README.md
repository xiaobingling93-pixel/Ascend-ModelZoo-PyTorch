# Qwen-VL
# 简介
## 模型介绍

**Qwen-VL** 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。Qwen-VL 可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。
本仓库主要将Qwen-VL模型的任务迁移到了昇腾NPU上，并进行极致性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|  模型  | 任务列表 | 是否支持 |
|:----:|:----:|:-----:|
| Qwen-VL |  训练  | ✔ |
| Qwen-VL | 在线推理 | ✔ |



## 代码实现

- 参考实现：

  ```
  url=https://github.com/QwenLM/Qwen-VL.git
  commit_id=aa00ed04091eea5fcdd32985e7915f1c53e7d599
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/foundation/Qwen-VL
  ```



## 准备训练环境

### 安装模型环境


  **表 1**  三方库版本支持表

  |     三方库     |  支持版本  |
  |:-----------:|:------:|
  |   PyTorch   | 2.1.0  |


   在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。


   ```python
   source ${cann_install_path}/ascend-toolkit/set_env.sh              # 激活cann环境
   cd Qwen-VL
   pip install -r requirements.txt                                    # 安装依赖
   ```

修改transformers源码：

- ./site-packages/transformers/training_args.py：屏蔽1410~1422行

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。
                

  **表 2**  昇腾软件版本支持表

  | 软件类型   |   支持版本   |
  | :--------: |:--------:|
  | FrameworkPTAdapter |   在研版本   |
  | CANN |   在研版本   |
  | 昇腾NPU固件 |   在研版本   | 
  | 昇腾NPU驱动 | 在研版本 |

  

### 准备数据集

#### 训练数据集准备

用户需自行制作数据集。参考Qwen-VL[官方指导资料](https://github.com/QwenLM/Qwen-VL/blob/master/README_CN.md)，将所有样本放到一个列表中并存入json文件中。每个样本对应一个字典，包含id和conversation，其中后者为一个列表。示例如下所示：

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是Qwen-VL,一个支持视觉输入的大模型。"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n图中的狗是什么品种？"
      },
      {
        "from": "assistant",
        "value": "图中是一只拉布拉多犬。"
      },
      {
        "from": "user",
        "value": "框出图中的格子衬衫"
      },
      {
        "from": "assistant",
        "value": "<ref>格子衬衫</ref><box>(588,499),(725,789)</box>"
      }
    ]
  },
  { 
    "id": "identity_2",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>assets/mm_tutorial/Chongqing.jpeg</img>\nPicture 2: <img>assets/mm_tutorial/Beijing.jpeg</img>\n图中都是哪"
      },
      {
        "from": "assistant",
        "value": "第一张图片是重庆的城市天际线，第二张图片是北京的天际线。"
      }
    ]
  }
]
```

为针对多样的VL任务，特殊tokens如下： `<img> </img> <ref> </ref> <box> </box>`.

- 对于带图像输入的内容可表示为 `Picture id: <img>img_path</img>\n{your prompt}`，其中`id`表示对话中的第几张图片。"img_path"可以是本地的图片或网络地址。 

- 对话中的检测框可以表示为`<box>(x1,y1),(x2,y2)</box>`，其中 `(x1, y1)` 和`(x2, y2)`分别对应左上角和右下角的坐标，并且被归一化到`[0, 1000)`的范围内. 检测框对应的文本描述也可以通过`<ref>text_caption</ref>`表示。



### 获取预训练模型

用户可访问huggingface或ModelScope网站获取Qwen-VL-Chat或者Qwen-VL模型权重。

将其中的模型代码`modeling_qwen.py`更换成本项目中的`./models/modeling_qwen.py`文件，`visual.py`替换成`./models/visual.py`



## 快速开始

### 训练任务

本任务主要提供**混精bf16**的**8卡**训练脚本。

#### 开始训练
1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行预训练脚本。

   该模型支持单机8卡训练。
   
  
   - 单机8卡训练
   
     ```shell 
     bash test/train_full_8p.sh --model_name=${预训练模型路径} --data_path=${训练数据集路径} # 8卡训练，混精bf16
     bash test/train_performance_8p.sh --model_name=${预训练模型路径} --data_path=${训练数据集路径}# 8卡性能，混精bf16
     ```
     
   - 模型训练python训练脚本参数说明如下。
   
   ```shell
   finetune.py
   --per_device_train_batch_size        //训练batch大小
   --gradient_accumulation_steps        //梯度累积大小  
   --num_train_epochs                   //设置训练轮数
   ```
   
#### 训练结果


##### 性能
| 芯片 | 卡数 | model_max_length | batch_size | gradient_accumulation_steps | AMP_Type | Torch_Version | tokens/p/s |
|:---:|:---:|:----:|:----------:|:---:|:---:|:---:|:---:|
| GPU | 8p | 2048 | 1 |     16     | bf16 | 2.1 | 1796 |
| Atlas A2 | 8p | 2048 | 1 |     16     | bf16 | 2.1 | 1910 |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 变更说明

## 变更

2024.04.29：首次发布

# FAQ

1. 如果出现SSL:CERTIFICATE_VERIFY_FAILED，可以手动将./site-packages/requests/adapters.py文件中，send函数中添加verify=False设置。



