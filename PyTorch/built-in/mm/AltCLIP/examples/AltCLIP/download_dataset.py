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
import os
from flagai.auto_model.auto_loader import AutoLoader
from torchvision.datasets import CIFAR10

auto_loader = AutoLoader(
    task_name="txt_img_matching",
    model_dir="./checkpoints",
    model_name="AltCLIP-XLMR-L"
)

tokenizer = auto_loader.get_tokenizer()
transform = auto_loader.get_transform()

dataset_root = "./clip_benchmark_datasets"
dataset_name = "cifar10"

train_dataset = CIFAR10(root=os.path.join(dataset_root, dataset_name), 
                transform=transform,   
                download=True)

test_dataset = CIFAR10(root=os.path.join(dataset_root, dataset_name), 
                transform=transform,
                train=False,   
                download=True)