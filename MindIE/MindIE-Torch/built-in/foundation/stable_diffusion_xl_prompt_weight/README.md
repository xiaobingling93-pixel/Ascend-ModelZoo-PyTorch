# stable-diffusionxl-prompt-weighting模型-推理指导  


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   stable-diffusionxl-prompt-weighting描述增强(类似++的操作)：通过“提示权重（prompt weighting）”来精细调控模型对输入文本提示中不同概念的关注程度，从而影响最终生成图像的内容和焦点。

- 参考实现：
  ```bash
   # stable-diffusionxl-prompt-weighting
   https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts#stable-diffusion-xl
  ```

- 设备支持：
Atlas 800I A2推理设备：支持的卡数为1
Atlas 300I Duo推理卡：支持的卡数为1

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表
  | 配套   | 版本    | 环境准备指导 |
  | ------ | ------- | ------------ |
  | Python | 3.10.13 | -            |
  | torch  | 2.1.0   | -            |     
该模型性能受CPU规格影响，建议使用64核CPU（arm）以复现性能


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。
   ```bash
   pip3 install -r requirements.txt
   ```

2. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```
   
3. 代码修改

   执行命令：
   
   ```bash
   python3 stable_diffusion_clip_patch.py
   ```
   
   ```bash
   python3 stable_diffusion_attention_patch.py
   ```

   ```bash
   # 若使用unetCache
   python3 stable_diffusionxl_unet_patch.py
   ```
   
## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型输入文本信息生成图片，无需数据集。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。
   使用Pytorch导出pt模型，然后使用MindIE推理引擎转换为适配昇腾的模型。

   0. 获取权重（可选）

      可提前下载权重，放到代码同级目录下，以避免执行后面步骤时可能会出现下载失败。

      ```bash
      # 需要使用 git-lfs (https://git-lfs.com)
      git lfs install

      # xl
      git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
      ```

   1. 导出pt模型并进行编译。(可选)
      
      ```bash
      # xl (执行时下载权重)
      model_base="stabilityai/stable-diffusion-xl-base-1.0"
      
      # xl (使用上一步下载的权重)
      model_base="./stable-diffusion-xl-base-1.0"
      ```

      执行命令：

      ```bash
      python3 export_ts_prompt_weight.py --model ${model_base} --output_dir ./models --batch_size 1 --flag 1 --soc A2 --device 0
      
      ```
      参数说明：
      - --model：模型权重路径
      - --output_dir: ONNX模型输出目录
      - --batch_size: 设置batch_size, 默认值为1,当前仅支持batch_size=1的场景
      - --flag：默认为1。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512。
      - --soc：只支持Duo和A2。默认为A2。
      - --device：推理设备ID
      - --use_cache: 【可选】在推理过程中使用cache
      
      静态编译场景：

      - ./models/clip/clip_bs{batch_size}.pt, ./models/clip/clip_bs{batch_size}_compile.ts 和 ./models/clip/clip2_bs{batch_size}.pt, ./models/clip/clip2_bs{batch_size}_compile.ts
      - ./models/unet/unet_bs{batch_size x 2}.pt, ./models/unet/unet_bs{batch_size x 2}_compile_static.ts
      - ./models/vae/vae_bs{batch_size}.pt, ./models/vae/vae_bs{batch_size}_compile_static.ts
      - ./models/ddim/ddim_bs{batch_size}.pt, ./models/ddim/ddim_bs{batch_size}_compile_static.ts

      动态分档场景：

      - ./models/clip/clip_bs{batch_size}.pt, ./models/clip/clip_bs{batch_size}_compile.ts 和 ./models/clip/clip2_bs{batch_size}.pt, ./models/clip/clip2_bs{batch_size}_compile.ts
      - ./models/unet/unet_bs{batch_size x 2}.pt, ./models/unet/unet_bs{batch_size x 2}_compile.ts
      - ./models/vae/vae_bs{batch_size}.pt, ./models/vae/vae_bs{batch_size}_compile.ts
      - ./models/ddim/ddim_bs{batch_size}.pt, ./models/ddim/ddim_bs{batch_size}_compile.ts
      
      
   
2. 开始推理验证。

   1. 执行推理脚本。
      ```bash
      # 不使用unetCache策略
      python3 stable_diffusionxl_pipeline_prompt_weight.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results \
              --steps 50 \
              --output_dir ./models \
              --flag 1 \
              --w_h 1024 
      
      # 使用UnetCache策略
      python3 stable_diffusionxl_pipeline_prompt_weight.py \
              --model ${model_base} \
              --prompt_file ./prompts.txt \
              --device 0 \
              --save_dir ./results_unetCache \
              --steps 50 \
              --output_dir ./models \
              --flag 1 \
              --w_h 1024 \
              --use_cache
      ```
      
      参数说明：
      - --model: 模型名称或本地模型目录的路径。
      - --prompt_file: 提示词文件。
      - --device: 推理设备ID。
      - --save_dir: 生成图片的存放目录。
      - --steps: 生成图片迭代次数。
      - --output_dir: 存放导出模型的目录。
      - --flag：默认为1。0代表静态，只支持分辨率为1024x1024；1代表动态分档，支持的分辨率为1024x1024和512x512。**注意**：请与导出模型时设置的flag保持一致
      - --w_h: image的宽高，设置为1024表示宽高均为1024，设置为512表示宽高均为512。仅支持这两种分辨率。
      - --use_cache: 【可选】在推理过程中使用cache。
      
      执行完成后在 `./results`目录下生成推理图片。

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。