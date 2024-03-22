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

# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
from .class_names import (ade_classes, ade_palette, bdd100k_classes,
                          bdd100k_palette, cityscapes_classes,
                          cityscapes_palette, cocostuff_classes,
                          cocostuff_palette, dataset_aliases, get_classes,
                          get_palette, isaid_classes, isaid_palette,
                          loveda_classes, loveda_palette, potsdam_classes,
                          potsdam_palette, stare_classes, stare_palette,
                          synapse_classes, synapse_palette, vaihingen_classes,
                          vaihingen_palette, voc_classes, voc_palette)
# yapf: enable
from .collect_env import collect_env
from .get_templates import get_predefined_templates
from .io import datafrombytes
from .misc import add_prefix, stack_batch
from .set_env import register_all_modules
from .tokenizer import tokenize
from .typing_utils import (ConfigType, ForwardResults, MultiConfig,
                           OptConfigType, OptMultiConfig, OptSampleList,
                           SampleList, TensorDict, TensorList)

# isort: off
from .mask_classification import MatchMasks, seg_data_to_instance_data
from .fcn_runner import FCNRunner

__all__ = [
    'collect_env',
    'register_all_modules',
    'stack_batch',
    'add_prefix',
    'ConfigType',
    'OptConfigType',
    'MultiConfig',
    'OptMultiConfig',
    'SampleList',
    'OptSampleList',
    'TensorDict',
    'TensorList',
    'ForwardResults',
    'cityscapes_classes',
    'ade_classes',
    'voc_classes',
    'cocostuff_classes',
    'loveda_classes',
    'potsdam_classes',
    'vaihingen_classes',
    'isaid_classes',
    'stare_classes',
    'cityscapes_palette',
    'ade_palette',
    'voc_palette',
    'cocostuff_palette',
    'loveda_palette',
    'potsdam_palette',
    'vaihingen_palette',
    'isaid_palette',
    'stare_palette',
    'dataset_aliases',
    'get_classes',
    'get_palette',
    'datafrombytes',
    'synapse_palette',
    'synapse_classes',
    'get_predefined_templates',
    'tokenize',
    'seg_data_to_instance_data',
    'MatchMasks',
    'bdd100k_classes',
    'bdd100k_palette',
]
