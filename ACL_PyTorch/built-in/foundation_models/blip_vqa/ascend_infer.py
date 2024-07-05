# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os
import time

from ais_bench.infer.interface import InferSession
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import yaml

from data.utils import pre_question
from models.blip_vqa import BLIP_VQA, tile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/vqa.yaml',
        help='Path of config file.',
    )
    parser.add_argument(
        '--result_file',
        type=str,
        default='blip_models/results.json',
        help='Path to save results.',
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./',
        help='Path of directory of images.',
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='blip_models',
        help='Path of directory to save ONNX models.',
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='NPU device id.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size of data loader.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of subprocesses to use for data loading.',
    )
    return parser.parse_args()


class BlipVQAInfer(BLIP_VQA):
    def __init__(
            self,
            med_config='configs/med_config.json',
            image_size=480,
            vit='base',
            vit_grad_ckpt=False,
            vit_ckpt_layer=0,
            model_dir='blip_models',
            device_id=0,
    ):
        super().__init__(
            med_config,
            image_size,
            vit,
            vit_grad_ckpt,
            vit_ckpt_layer
        )

        visual_encoder = os.path.join(model_dir, 'visual_encoder_md.om')
        text_encoder = os.path.join(model_dir, 'text_encoder_md.om')
        text_decoder_1 = os.path.join(model_dir, 'text_decoder_1_md.om')
        text_decoder_2 = os.path.join(model_dir, 'text_decoder_2_md.om')

        self.visual_encoder_om = InferSession(device_id, visual_encoder)
        self.text_encoder_om = InferSession(device_id, text_encoder)
        self.text_decoder_om_1 = InferSession(device_id, text_decoder_1)
        self.text_decoder_om_2 = InferSession(device_id, text_decoder_2)

    def forward(self, image, question, answer=None, k_test=128):
        image_embeds = torch.from_numpy(
            self.visual_encoder_om.infer([image.numpy()])[0]
        )

        question = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors='pt'
        )
        question.input_ids[:, 0] = self.tokenizer.enc_token_id

        question_states = torch.from_numpy(
            self.text_encoder_om.infer(
                [
                    question.input_ids.numpy(),
                    question.attention_mask.numpy(),
                    image_embeds.numpy(),
                ]
            )[0]
        )

        max_ids = self.rank_answer(
            question_states,
            question.attention_mask,
            answer.input_ids,
            answer.attention_mask,
            k_test
        )

        return max_ids

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token
        logits = torch.from_numpy(
            self.text_decoder_om_1.infer(
                [
                    start_ids.numpy(),
                    question_states.numpy(),
                    question_atts.numpy()
                ]
            )[0]
        )

        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        loss = torch.from_numpy(
            self.text_decoder_om_2.infer(
                [
                    input_ids.numpy(),
                    input_atts.numpy(),
                    question_states.numpy(),
                    question_atts.numpy(),
                    targets_ids.numpy(),
                ]
            )[0]
        )

        log_probs_sum = -loss
        log_probs_sum = log_probs_sum.view(num_ques, k)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
        return max_ids


class VQADataset(Dataset):
    def __init__(self, config, image_dir):
        """
        config (dict): config of model
        image_dir (string): image root
        """
        ann_root = config['ann_root']
        download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json', ann_root)
        self.annotation = json.load(open(os.path.join(ann_root, 'vqa_test.json'), 'r'))

        download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json', ann_root)
        self.answer_list = json.load(open(os.path.join(ann_root, 'answer_list.json'), 'r'))

        self.transform = transforms.Compose([
            transforms.Resize(
                (config['image_size'], config['image_size']),
                interpolation=InterpolationMode.BICUBIC
            ),
        ])

        self.image_root = image_dir

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = np.asarray(image)

        question = pre_question(ann['question'])
        question_id = ann['question_id']

        return image, question, question_id


def main(args):
    config = yaml.safe_load(open(args.config, 'r'))

    #### Model ####
    model = BlipVQAInfer(
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
        model_dir=args.model_dir,
        device_id=args.device
    )
    model.eval()

    #### Dataset ####
    dataset = VQADataset(config, args.image_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    answer_list = dataset.answer_list
    answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt')
    answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id

    results = []
    start_time = time.time()
    for image, question, question_id in tqdm(data_loader):
        answer_ids = model(image, question, answer_candidates, k_test=config['k_test'])

        for ques_id, answer_id in zip(question_id, answer_ids):
            results.append({"question_id": int(ques_id.item()), "answer": answer_list[answer_id]})

    total_time = time.time() - start_time
    print('[performance], total_time: %.4f' % total_time)
    print('[performance], data number:', len(results))
    print('[performance], average: %.4f data/s' % (len(results) / total_time))

    flags = os.O_CREAT | os.O_WRONLY
    with os.fdopen(os.open(args.result_file, flags, 0o755), 'w') as save_file:
        json.dump(results, save_file)

    print('Result file saved to %s' % args.result_file)


if __name__ == '__main__':
    main(parse_args())
