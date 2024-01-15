# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import json
import time
import glob
import argparse
from tqdm import tqdm
from PIL import Image

import yaml
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import InterpolationMode
from ais_bench.infer.interface import InferSession

from models.blip import blip_decoder
from data.utils import save_result, coco_caption_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str, 
        default='./configs/caption_coco.yaml',
        help='Path of config file.',
    )
    parser.add_argument(
        '--caption_file', 
        type=str, 
        default='blip_models/captions.json',
        help='Path to save generated captions.',
    )
    parser.add_argument(
        '--dataset_split', 
        choices=['val', 'test'], 
        default='val',
        help='Dataset used to evaluate the model.',
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
        '--mode', 
        choices=['dymshape', 'dymdims'], 
        default='dymdims',
        help='Mode of decoder om inputs.',
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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the generated captions.",
    )
    parser.add_argument(
        '--evaluate_file', 
        type=str, 
        default='blip_models/evaluate.json',
        help='Path to save the file of evaluate result.',
    )
    return parser.parse_args()


class CocoCaptionDataset(Dataset):
    def __init__(self, config, split):
        """
        config (dict): config of model
        split (string): val or test
        """
        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {
            'val': 'coco_karpathy_val.json',
            'test': 'coco_karpathy_test.json'
        }
        
        download_url(urls[split], config['ann_root'])
        file_path = os.path.join(config['ann_root'], filenames[split])
        
        self.annotation = json.load(open(file_path, 'r'))
        self.transform = transforms.Compose([
            transforms.Resize(
                (config['image_size'], config['image_size']),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        
        ann = self.annotation[index]
        
        image = Image.open(ann['image']).convert('RGB')
        image = self.transform(image)
        
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)


def create_dataloader(dataset, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def generate(args, config):
    model = blip_decoder(
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
        prompt=config['prompt']
    )
    model.eval()

    visual_encoder = os.path.join(args.model_dir, 'visual_encoder.om')
    if args.mode == 'dymdims':
        text_decoder = os.path.join(args.model_dir, 'text_decoder.om')
    else:
        decoder_path = os.path.join(args.model_dir, 'text_decoder_*.om')
        text_decoder = glob.glob(decoder_path)[0]

    model.visual_encoder.om = InferSession(args.device, visual_encoder)
    model.text_decoder.om = InferSession(args.device, text_decoder)
    model.text_decoder.om.mode = args.mode

    dataset = CocoCaptionDataset(config, args.dataset_split)
    data_loader = create_dataloader(
        dataset,
        args.batch_size,
        args.num_workers,
    )

    results = []
    start_time = time.time()
    for image, image_id in tqdm(data_loader):
        data_num = image_id.shape[0]
        if data_num < args.batch_size:
            shape = list(image.size())
            shape[0] = args.batch_size - data_num
            padding_data = torch.zeros(shape, dtype=image.dtype)
            image = torch.cat((image, padding_data), 0)
        captions = model.generate(
            image, sample=False, num_beams=config['num_beams'], 
            max_length=config['max_length'], min_length=config['min_length']
        )
        captions = captions[:data_num]
        for caption, img_id in zip(captions, image_id):
            results.append({'image_id': img_id.item(), 'caption': caption})

    total_time = time.time() - start_time
    print('[performance], total_time: %.4f' % total_time)
    print('[performance], data number:', len(results))
    print('[performance], average: %.4f image/s' % (len(results) / total_time))

    flags = os.O_CREAT | os.O_WRONLY
    with os.fdopen(os.open(args.caption_file, flags, 0o755), 'w') as save_file:
        json.dump(results, save_file)

    print('Result file saved to %s' % args.caption_file)

    return


def evaluate(reference, caption_file, evaluate_file, dataset):
    coco_score = coco_caption_eval(
        reference,
        caption_file, 
        dataset
    )

    with os.fdopen(os.open(evaluate_file, flags, 0o755), 'w') as save_file:
        json.dump(coco_score.eval, save_file)
    print('Evaluate file saved to %s' % evaluate_file)

    return


def main(args):
    config = yaml.safe_load(open(args.config, 'r'))

    if args.evaluate:
        if not os.path.exists(args.caption_file):
            print('The caption file does not exist! Generating captions...')
            generate(args, config)

        evaluate(
            config['coco_gt_root'], 
            args.caption_file, 
            args.evaluate_file, 
            args.dataset_split
        )
    
    else:
        generate(args, config)


if __name__ == '__main__':
    main(parse_args())
