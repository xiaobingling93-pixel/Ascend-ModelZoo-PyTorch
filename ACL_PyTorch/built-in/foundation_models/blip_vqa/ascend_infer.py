#!/usr/bin/env python3
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
import json
import os
import pathlib
import time

from ais_bench.infer import interface as infer_interface
from torch.utils import data as torch_data
from torchvision import transforms
from torchvision.transforms import functional
import tqdm
import yaml

from models import blip_vqa
from data import vqa_dataset
# `vqa_dataset` imported is a class rather than a module, because `from data.vqa_dataset import vqa_dataset` is executed
# at the beginning of data/__init__.py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vqa.yaml',
        help='Path of config file.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
        help='Batch size of data loader. Should be consistent with the batch size of the OM model.',
    )
    parser.add_argument(
        '--infer_mode',
        choices=['rank', 'generate'],
        default='rank',
        help='Mode of inference.',
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='blip_models',
        help='Path of directory to save ONNX models.',
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='.',
        help='Path of directory of images.',
    )
    parser.add_argument(
        '--result_file',
        type=str,
        default='ascend_infer_results.json',
        help='Path to save results.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of subprocesses to use for data loading.',
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='NPU device id.',
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # The following code is adapted from
    # https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/train_vqa.py

    config = yaml.safe_load(open(args.config))

    model = blip_vqa.blip_vqa(
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
    )

    visual_encoder_path = pathlib.Path(args.model_dir, 'visual_encoder_md.om').as_posix()
    model.visual_encoder.om = infer_interface.InferSession(args.device, visual_encoder_path)
    text_encoder_path = pathlib.Path(args.model_dir, 'text_encoder_md.om').as_posix()
    model.text_encoder.om = infer_interface.InferSession(args.device, text_encoder_path)
    if args.infer_mode == 'rank':
        text_decoder_rank_1_path = pathlib.Path(args.model_dir, 'text_decoder_rank_1_md.om').as_posix()
        model.text_decoder.rank_1_om = infer_interface.InferSession(args.device, text_decoder_rank_1_path)
        text_decoder_rank_2_path = pathlib.Path(args.model_dir, 'text_decoder_rank_2_md.om').as_posix()
        model.text_decoder.rank_2_om = infer_interface.InferSession(args.device, text_decoder_rank_2_path)
    elif args.infer_mode == 'generate':
        text_decoder_generate_path = pathlib.Path(args.model_dir, 'text_decoder_generate_sim.om').as_posix()
        model.text_decoder.generate_om = infer_interface.InferSession(args.device, text_decoder_generate_path)

    transform = transforms.Compose([
        transforms.Resize(
            (config['image_size'], config['image_size']),
            interpolation=functional.InterpolationMode.BICUBIC,
        ),
    ])
    dataset = vqa_dataset(transform, config['ann_root'], args.image_dir, config['vg_root'], split='test')
    data_loader = torch_data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    results = []
    if args.infer_mode == 'rank':
        answer_list = dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt')
        answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id
        start_time = time.time()
        for image_path, image, question, question_id in tqdm.tqdm(data_loader):
            answer_ids = model(
                image,
                question,
                answer=answer_candidates,
                train=False,
                inference=args.infer_mode,
                k_test=config['k_test'],
            )
            for i in range(args.batch_size):
                results.append({
                    'image_path': image_path[i],
                    'question': question[i],
                    'question_id': int(question_id[i]),
                    'answer': answer_list[answer_ids[i]],
                })
        total_time = time.time() - start_time
        print(f'[performance], total_time: {total_time:.4f}')
        print(f'[performance], data number: {len(results)}')
        print(f'[performance], average: {len(results) / total_time:.4f} data/s')
    elif args.infer_mode == 'generate':
        for image_path, image, question, question_id in tqdm.tqdm(data_loader):
            answers = model(image, question, train=False, inference=args.infer_mode, k_test=config['k_test'])
            for i in range(args.batch_size):
                results.append({
                    'image_path': image_path[i],
                    'question': question[i],
                    'question_id': question_id[i].item(),
                    'answer': answers[i],
                })

    flags = os.O_CREAT | os.O_WRONLY
    with os.fdopen(os.open(args.result_file, flags, 0o755), 'w') as save_file:
        json.dump(results, save_file, indent=4)

    print('Result file saved to %s' % args.result_file)


if __name__ == '__main__':
    main(_parse_args())
