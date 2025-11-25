# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd
# This software is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import json
import re
import numpy as np
import argparse

from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer



def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def preprocess(src_path, img_path, ids_path, mask_path):

    ann_root = 'annotation'
    if not os.path.exists(ann_root):
        os.mkdir(ann_root)
    
    filenames = {'val': 'coco_karpathy_val.json',
                 'test': 'coco_karpathy_test.json'}
    annotation = json.load(
        open(os.path.join(ann_root, filenames['test']), 'r'))

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        normalize, ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    txt_id = 0
    for img_id, ann in enumerate(annotation):
        image = ann['image']
        input_image = Image.open(os.path.join(src_path, image)).convert('RGB')
        input_tensor = transform_test(input_image)
        img_np = np.array(input_tensor).astype(np.float32)
        img_np.tofile(os.path.join(img_path, str(img_id) + ".bin"))

        for i, caption in enumerate(ann['caption']):
            txt = pre_caption(caption, 30)
            txt = tokenizer(txt, padding='max_length',
                            truncation=True, max_length=35, return_tensors="pt")
            input_ids = txt.input_ids
            input_mask = txt.attention_mask
            input_ids_np = input_ids.numpy().astype(np.int32)
            input_mask_np = input_mask.numpy().astype(np.int32)
            input_ids_np.tofile(os.path.join(
                ids_path, str(txt_id) + '.bin'))
            input_mask_np.tofile(os.path.join(
                mask_path, str(txt_id) + '.bin'))
            txt_id += 1

    print("generate bin runs successfuly!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', default='/opt/npu/dcc/coco2014/')
    parser.add_argument('--save_bin_path',
                        default='/opt/npu/dcc/coco2014_bin/')
    args = parser.parse_args()

    coco_path = args.coco_path
    save_bin_path = args.save_bin_path

    save_img_path = os.path.join(save_bin_path, 'img/')
    save_ids_path = os.path.join(save_bin_path, 'ids/')
    save_mask_path = os.path.join(save_bin_path, 'mask/')

    for path in [save_ids_path, save_img_path, save_mask_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    preprocess(coco_path, save_img_path, save_ids_path, save_mask_path)
