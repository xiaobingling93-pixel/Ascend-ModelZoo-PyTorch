# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd
# This software is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import re
import json
import argparse

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np

import utils as utils
from models.blip_itm import blip_itm


@torch.no_grad()
def evaluation(image_bin_path, image_feat_bin_path, text_bin_path, ids_path, mask_path, model, max_samples=None, device_str='npu:0'):
    k_test = 20

    print("k_test: ", k_test)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    device = torch.device(device_str)
    print("Running on : ", device)

    model.to(device)
    model.eval()

    image_embed_files = os.listdir(image_bin_path)
    image_feat_files = os.listdir(image_feat_bin_path)
    text_embed_files = os.listdir(text_bin_path)


    # In order by number but not str, such as 0.bin, 1.bin, 2.bin, ..., 10.bin, 11.bin
    image_embed_files.sort(key=lambda x: int(x.split('.')[0]))
    image_feat_files.sort(key=lambda x: int(x.split('.')[0]))
    text_embed_files.sort(key=lambda x: int(x.split('.')[0]))

    all_text_ids = []
    text_id_dic = {}
    if max_samples is not None:
        # Load annotation file to make sure the dictionary of image-text.
        ann_root = 'annotation'
        filenames = {'test': 'coco_karpathy_test.json'}
        annotation = json.load(open(os.path.join(ann_root, filenames['test']), 'r'))

        # Build dictionary of image id -> text ids.
        text_id_counter = 0
        for img_id, ann in enumerate(annotation):
            if img_id >= max_samples:
                break
            caption_count = len(ann['caption'])
            text_id_dic[img_id] = list(range(text_id_counter, text_id_counter + caption_count))
            text_id_counter += caption_count

        # Collect all the text ids that need to be processed
        for text_ids in text_id_dic.values():
            all_text_ids.extend(text_ids)

        image_embed_files = image_embed_files[:max_samples]
        image_feat_files = image_feat_files[:max_samples]
        text_embed_files = [text_embed_files[i] for i in all_text_ids if i < len(text_embed_files)]

        print(f"Limit the number of samples: The first {max_samples} images correspond to {len(text_embed_files)} texts")

    # read bin files
    text_embeds = []  # [256]
    image_embeds = []  # [256]
    image_feats = []  # [577*768]

    for file in image_embed_files:
        file_path = os.path.join(image_bin_path, file)
        image_embed = np.fromfile(file_path, dtype=np.float32)  # [256]
        image_embeds.append(image_embed)

    for file in image_feat_files:
        file_path = os.path.join(image_feat_path, file)
        image_feat = np.fromfile(file_path, dtype=np.float32)  # [443136]
        image_feat = image_feat.reshape(577, 768)
        image_feats.append(image_feat)

    for file in text_embed_files:
        file_path = os.path.join(text_bin_path, file)
        text_embed = np.fromfile(file_path, dtype=np.float32)  # [256]
        text_embeds.append(text_embed)

    print("Load bins completed.")

    image_embeds = torch.tensor(np.array(image_embeds))
    image_feats = torch.tensor(np.array(image_feats), device=device)
    text_embeds = torch.tensor(np.array(text_embeds))
    print("convert bins to tensors completed")

    text_atts = []
    ids_dir = os.listdir(ids_path)
    mask_dir = os.listdir(mask_path)
    ids_dir.sort(key=lambda x: int(x.split('.')[0]))
    mask_dir.sort(key=lambda x: int(x.split('.')[0]))
    text_ids = []

    for i, ids_bin in enumerate(ids_dir):
        if max_samples is not None and i not in all_text_ids:
            continue
        files = np.fromfile(ids_path + ids_bin, dtype=np.int32)
        text_ids.append(files)

    for i, atts_bin in enumerate(mask_dir):
        if max_samples is not None and i not in all_text_ids:
            continue
        files = np.fromfile(mask_path + atts_bin, dtype=np.int32)
        text_atts.append(files)

    print("load text ids and mask completed.")

    text_ids = torch.tensor(np.array(text_ids), device=device)
    text_atts = torch.tensor(np.array(text_atts), device=device)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    print("convert text ids and masks to tensor completed.")
    print("text_ids, text_atts:", text_ids.shape, text_atts.shape)

    sims_matrix = image_embeds @ text_embeds.t()
    sims_matrix = sims_matrix.to(device)
    score_matrix_i2t = torch.full(
        (len(image_embed_files), len(text_embed_files)), -100.0).to(device)
    print("### begin calcute score_matrix_i2t")

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[
                                 :-1], dtype=torch.long).to(device)

        output = model.text_encoder(text_ids[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score + topk_sim

    print("### end calcute score_matrix_i2t")

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(text_embed_files), len(image_embed_files)), -100.0)
    score_matrix_t2i = score_matrix_t2i.to(device)

    print("### begin calcute score_matrix_t2i")

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[
                                 :-1], dtype=torch.long).to(device)

        output = model.text_encoder(text_ids[i].repeat(k_test, 1).to(device),
                                    attention_mask=text_atts[i].repeat(k_test, 1).to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score + topk_sim

    print("### end calcute score_matrix_t2i")

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def itm_eval(scores_i2t, scores_t2i, text2image, image2text):
    print("begin calcute matrics.")
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in image2text[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == text2image[index])[0][0]
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    eval_result = {'text_r1': tr1,
                   'text_r5': tr5,
                   'text_r10': tr10,
                   'text_r_mean': tr_mean,
                   'image_r1': ir1,
                   'image_r5': ir5,
                   'image_r10': ir10,
                   'image_r_mean': ir_mean,
                   'r_mean': r_mean}
    print('eval_result:', eval_result)
    return eval_result


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_embed_path',
                        default='./coco2014_infer/text_embed/xxx/')
    parser.add_argument('--image_embed_path',
                        default='./coco2014_infer/image_embed/xxx/')
    parser.add_argument('--image_feat_path',
                        default='./coco2014_infer/image_feat/xxx/')
    parser.add_argument('--coco_bin_path', default='./coco2014_bin/')
    parser.add_argument('--pth_path', default='/model_base_retrieval_coco.pth')
    parser.add_argument('--max_samples', type=int, default=None, help='number of samples')
    parser.add_argument('--device', default=0, type=int, help='npu device to use (0, 1, 2, etc.)')
    args = parser.parse_args()
    args.device = f"npu:{str(args.device)}"

    print("running calculate metrics...")

    image_embed_path = args.image_embed_path
    text_embed_path = args.text_embed_path
    image_feat_path = args.image_feat_path

    ids_path = os.path.join(args.coco_bin_path, 'ids/')
    mask_path = os.path.join(args.coco_bin_path, 'mask/')

    pretrained_model = args.pth_path
    model = blip_itm(pretrained=pretrained_model)
    with torch.no_grad():
        score_matrix_i2t, score_matrix_t2i = evaluation(image_embed_path, image_feat_path, text_embed_path, ids_path, mask_path, model, args.max_samples, args.device)

    ann_root = 'annotation'
    filenames = {'val': 'coco_karpathy_val.json',
                 'test': 'coco_karpathy_test.json'}
    annotation = json.load(
        open(os.path.join(ann_root, filenames['test']), 'r'))

    text = []
    image = []
    text2image = {}
    image2text = {}

    text_id = 0
    for image_id, ann in enumerate(annotation):
        image.append(ann['image'])
        image2text[image_id] = []
        for i, caption in enumerate(ann['caption']):
            text.append(pre_caption(caption, 30))
            image2text[image_id].append(text_id)
            text2image[text_id] = image_id
            text_id += 1

    print("prepare test data completed.")

    itm_eval(score_matrix_i2t, score_matrix_t2i, text2image, image2text)