#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import json
import inspect
from .file_utils import standardize_path, safe_open, SAFEOPEN_FILE_PERMISSION


class ConfigMixin:
    r"""
    The `ConfigMixin` class is used to load the configuration from the file named `config_name`.

    Args:
        config_name (str): The name of the configuration file.
        model_path (str): The path to the configuration directory.
        **kwargs: Additional keyword arguments for the configuration.
    """
    config_name = None

    @classmethod
    def load_config(cls, model_path, **kwargs):
        r"""
        The class method is used to load the configuration from the file named `config_name`.

        Args:
            model_path (str): The path to the configuration directory.
            **kwargs: Additional keyword arguments for the configuration.

        Returns:
            A tuple containing the initialized dictionary and the extra configuration dictionary.
        """
        if cls.config_name is None:
            raise ValueError("Attr config_name is none.")

        model_path = standardize_path(model_path)
        config_path = os.path.join(model_path, cls.config_name)
        config_dict = _load_json_dict(config_path)

        # Get all required parameters
        all_parameters = inspect.signature(cls.__init__).parameters

        init_keys = set(dict(all_parameters))
        init_keys.discard("self")
        init_keys.discard('kwargs')

        init_dict = {}
        for key in init_keys:
            # If key in config, use config
            if key in config_dict:
                init_dict[key] = config_dict.pop(key)
            # If key in kwargs, use kwargs, this may rewrite config_dict
            if key in kwargs:
                init_dict[key] = kwargs.pop(key)
        return init_dict, config_dict

    def _init(self, value):
        init_signature = inspect.signature(self.__init__)
        parameters = init_signature.parameters
        for param_name, _ in parameters.items():
            if param_name != 'self':
                setattr(self, param_name, value.get(param_name, None))


def _load_json_dict(config_path):
    with safe_open(config_path, "r", encoding="utf-8", permission_mode=SAFEOPEN_FILE_PERMISSION) as reader:
        data = reader.read()
    return json.loads(data, strict=False)