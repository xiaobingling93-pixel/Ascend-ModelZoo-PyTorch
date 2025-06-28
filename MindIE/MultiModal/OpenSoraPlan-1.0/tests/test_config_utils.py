#!/usr/bin/env python
# coding=utf-8
# Copyright(C) 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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
# limitations under the License

import unittest
import logging
import sys
import os
import json
sys.path.append('../')
from mindiesd.config_utils import ConfigMixin

logger = logging.getLogger()
logger.setLevel(logging.INFO)


USED_KEY = "used_key"
NOUSED_KEY = "noused_key"
CONFIG_NAME = "./configs"


class ModelConfig(ConfigMixin):
    config_name = "configmixin.json"

    def __init__(self, used_key):
        self.used_key = used_key


class InvalidModelConfig(ConfigMixin):
    config_name = "invalid.json"

    def __init__(self, used_key):
        self.used_key = used_key


class TestConfigMixin(unittest.TestCase):

    def test_load_config(self):
        init_dict, config_dict = ModelConfig.load_config(CONFIG_NAME)
        # used_key will in init_dict
        self.assertIn(USED_KEY, init_dict)
        self.assertEqual(init_dict.get(USED_KEY), USED_KEY)
        self.assertNotIn(USED_KEY, config_dict)

        # noused_key will in config_dict
        self.assertIn(NOUSED_KEY, config_dict)
        self.assertEqual(config_dict.get(NOUSED_KEY), NOUSED_KEY)
        self.assertNotIn(NOUSED_KEY, init_dict)

    def test_config_path_invalid(self):
        try:
            init_dict, config_dict = ModelConfig.load_config("./no_used_path")
        except Exception as e:
            logger.error(e)
            init_dict, config_dict = None, None
        self.assertIsNone(init_dict)
        self.assertIsNone(config_dict)
    
    def test_config_path_none(self):
        try:
            init_dict, config_dict = ConfigMixin.load_config(CONFIG_NAME)
        except Exception as e:
            logger.error(e)
            init_dict, config_dict = None, None
        self.assertIsNone(init_dict)
        self.assertIsNone(config_dict)
    
    def test_config_json_invalid(self):
        try:
            init_dict, config_dict = InvalidModelConfig.load_config(CONFIG_NAME)
        except Exception as e:
            logger.error(e)
            init_dict, config_dict = None, None
        self.assertIsNone(init_dict)
        self.assertIsNone(config_dict)
        

if __name__ == '__main__':
    unittest.main()