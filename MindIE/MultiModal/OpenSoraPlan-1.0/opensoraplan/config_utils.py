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

import os
import json
import inspect
from typing import Dict, Tuple
from .utils.log import logger


class ConfigMixin:
    config_name = None

    @classmethod
    def load_config(cls, model_path, **kwargs) -> Tuple[Dict, Dict]:
        if cls.config_name is None:
            logger.error("config_name is not defined.")
            raise ValueError("config_name is not defined.")

        if model_path is None:
            logger.error("model_path must not be None")
            raise ValueError("model_path must not be None")

        model_path = os.path.abspath(model_path)
        config_path = os.path.join(model_path, cls.config_name)
        if not (os.path.exists(config_path) and os.path.isfile(config_path)):
            logger.error("%s is not found in %s!", cls.config_name, model_path)
            raise ValueError("%s is not found in %s!" % (cls.config_name, model_path))

        config_dict = _load_json_dict(config_path)

        # get all required parameters
        all_parameters = inspect.signature(cls.__init__).parameters

        init_keys = set(dict(all_parameters))
        init_keys.remove("self")
        if 'kwargs' in init_keys:
            init_keys.remove('kwargs')

        init_dict = {}
        for key in init_keys:
            # if key in config, use config
            if key in config_dict:
                init_dict[key] = config_dict.pop(key)
            # if key in kwargs, use kwargs, this may rewrite config_dict
            if key in kwargs:
                init_dict[key] = kwargs.pop(key)
        in_keys = set(init_dict.keys())
        if len(init_keys - in_keys) > 0:
            logger.warning("%s was not found in config and kwargs! Use default values.", init_keys - in_keys)
        return init_dict, config_dict


def _load_json_dict(config_path):
    with open(config_path, "r", encoding="utf-8") as reader:
        data = reader.read()
    return json.loads(data, strict=False)