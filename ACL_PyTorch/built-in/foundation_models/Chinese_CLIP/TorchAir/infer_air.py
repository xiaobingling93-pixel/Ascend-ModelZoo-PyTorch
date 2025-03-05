# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
from time import time

import torch
from PIL import Image
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

import cn_clip.clip as clip
from cn_clip.clip import load_from_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chinese Clip Inference")
    parser.add_argument(
        "--model_root_path",
        type=str,
        required=True,
        help="Dir of Pretrained Weight"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ViT-B-16",
        choices=['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50'],
        help="Type of Model"
    )
    parser.add_argument(
        '--image',
        type=str,
        default="examples/pokemon.jpeg",
        help='Path of Input Image File'
    )
    parser.add_argument('--text', type=str, default="杰尼龟", help='Input Text')
    parser.add_argument('--length', type=int, default=64, help='Context Length of Tokenizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--device', type=int, default=0, help='Npu Device Id')
    parser.add_argument('--loop', default=10, type=int, help='loop time')
    args = parser.parse_args()

    # adapt torchair
    device = torch.device('npu:{}'.format(args.device))
    torch_npu.npu.set_device(args.device)
    model, preprocess = load_from_name(args.model_type, device=device, download_root=args.model_root_path)
    model.eval()
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backbend = tng.get_npu_backend(compiler_config=config)
    model.visual = torch.compile(model.visual, dynamic=False, fullgraph=True, backend=npu_backbend)
    tng.use_internal_format_weight(model.visual)
    model.bert = torch.compile(model.bert, dynamic=False, fullgraph=True, backend=npu_backbend)
    tng.use_internal_format_weight(model.bert)

    # data preprocess
    image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
    text = clip.tokenize([args.text], context_length=args.length).to(device)
    image = image.repeat(args.batch_size, 1, 1, 1)  # bs, 3, 224, 224
    text = text.repeat(args.batch_size, 1)   # bs, length

    with torch.no_grad():
        # compile and precision test
        print('start compiling...')
        logits_per_image, logits_per_text = model.get_similarity(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs[0, 0])

        # test bert model performance
        pad_index = model.tokenizer.vocab['[PAD]']
        attn_mask = ~text.ne(pad_index)
        text_times = []
        for i in range(args.loop):
            torch.npu.synchronize()
            st = time()
            model.bert(text, attention_mask=attn_mask)
            torch.npu.synchronize()
            text_times.append(time() - st)
        text_perf = round(args.batch_size / (sum(text_times) / args.loop), 2)
        # test visual model performance
        image = image.type(model.dtype)
        image_times = []
        for i in range(args.loop):
            torch.npu.synchronize()
            st = time()
            model.visual(image, 0)
            torch.npu.synchronize()
            image_times.append(time() - st)
        image_perf = round(args.batch_size / (sum(image_times) / args.loop), 2)

        print(f'visual model performance: {image_perf} image/s')
        print(f'bert model performance: {text_perf} text/s')
