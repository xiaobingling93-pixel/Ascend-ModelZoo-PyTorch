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
import os
import io
import shutil
import tarfile
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
import wget

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest


def download_an4(target_dir: str,
                 manifest_dir: str,
                 min_duration: float,
                 max_duration: float,
                 val_fraction: float,
                 sample_rate: int):
    raw_tar_path = 'an4.tar.gz'
    if not os.path.exists(raw_tar_path):
        wget.download('https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4.tar.gz')
    tar = tarfile.open('an4.tar.gz')
    os.makedirs(target_dir, exist_ok=True)
    tar.extractall(target_dir)
    
    train_path = os.path.join(target_dir, 'train/')
    val_path = os.path.join(target_dir, 'val/')
    test_path = os.path.join(target_dir, 'test/')

    print('Creating manifests...')
    create_manifest(data_path=train_path,
                    output_name='an4_train_manifest.csv',
                    manifest_path=manifest_dir,
                    min_duration=min_duration,
                    max_duration=max_duration)
    create_manifest(data_path=val_path,
                    output_name='an4_val_manifest.csv',
                    manifest_path=manifest_dir,
                    min_duration=min_duration,
                    max_duration=max_duration)
    create_manifest(data_path=test_path,
                    output_name='an4_test_manifest.csv',
                    manifest_path=manifest_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes and downloads an4.')
    parser = add_data_opts(parser)
    parser.add_argument('--target-dir', default='an4_dataset/', help='Path to save dataset')
    parser.add_argument('--val-fraction', default=0.1, type=float,
                        help='Number of files in the training set to use as validation.')
    args = parser.parse_args()
    download_an4(target_dir=args.target_dir,
                 manifest_dir=args.manifest_dir,
                 min_duration=args.min_duration,
                 max_duration=args.max_duration,
                 val_fraction=args.val_fraction,
                 sample_rate=args.sample_rate)
