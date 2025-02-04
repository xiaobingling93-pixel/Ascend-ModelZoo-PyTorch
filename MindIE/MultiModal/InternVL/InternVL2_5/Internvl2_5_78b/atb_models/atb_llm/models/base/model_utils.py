# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Part of this file was copied from project transformers 4.29.0
import collections
import json
import os
import re
from typing import Union, Callable, Optional, Type, Any
from dataclasses import dataclass

import torch
import transformers
from torch import nn

from safetensors.torch import save_file as safe_save_file
from transformers.modeling_utils import shard_checkpoint
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig

from atb_llm.utils.weights import QUANTIZE_DTYPE_LIST
from atb_llm.utils import file_utils

WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_NAME = "model.safetensors"


def disable_logger_decorator(func: Callable):
    """Set logger.disable to True in the wrapped function."""
    def wrapper(name):
        logger = func(name)
        logger.disabled = True
        return logger
    return wrapper

transformers.utils.logging.get_logger = disable_logger_decorator(transformers.utils.logging.get_logger)


def unwrap_model_state_dict(state_dict: dict) -> dict:
    """Remove the prefix 'model.' from the keys of the state dict."""
    new_state_dict = {}
    for name, tensor in state_dict.items():
        new_name = name.replace('.linear.', '.')
        new_state_dict[new_name] = tensor
    return new_state_dict


class BaseModel(nn.Module):
    """Base class for all models."""
    def save_pretrained(self,
                        save_directory: str,
                        max_shard_size: Union[int, str] = "10GB",
                        save_function: Callable = torch.save,
                        safe_serialization: bool = False):
        """
        Save a model and its configuration file to `save_directory`.

        Args:
            save_directory (str): Directory to which to save, will be created if it doesn't exist.
            max_shard_size (Union[int, str], optional): Maximum size for an individual saved shard file.
                If the model size is larger than this ,the model will be split into multiple shards.
                Can be an integer (in bytes) or string with units ('42GB' or '42MB'), defaults to `10GB`.
            save_function (Callable, optional): The function to use to save the model files.
                Defaults to `torch.save`.
            safe_serialization (bool): Whether to use the safe serialization method, defaults to `Fasle`.
        """
        os.makedirs(save_directory, exist_ok=True)
        save_directory = file_utils.standardize_path(save_directory, check_link=False)
        file_utils.check_path_permission(save_directory)
        state_dict = unwrap_model_state_dict(self.state_dict())
        if safe_serialization:
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                ident = (tensor.data_ptr(), tensor.device, tensor.shape, tensor.stride())
                ptrs[ident].append(name)

            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
            warn_names = set()
            for names in shared_ptrs.values():
                found = 0
                for name in names:
                    if name in state_dict:
                        found += 1
                        if found > 1:
                            del state_dict[name]
                            warn_names.add(name)
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

            shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

            for filename in file_utils.safe_listdir(save_directory):
                full_filename = os.path.join(save_directory, filename)
                weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
                filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
                reg = re.compile(r"(.{1,4096})-\d{5}-of-\d{5}")

                need_remove = (filename.startswith(weights_no_suffix) and
                               os.path.isfile(full_filename) and
                               filename not in shards.keys() and
                               reg.fullmatch(filename_no_suffix) is not None)
                if need_remove:
                    full_filename = file_utils.standardize_path(full_filename)
                    file_utils.check_file_safety(full_filename, 'r', is_check_file_size=False)
                    os.remove(full_filename)
            for shard_file, shard in shards.items():
                save_shard_file = os.path.join(save_directory, shard_file)
                save_shard_file = file_utils.standardize_path(save_shard_file)
                file_utils.check_file_safety(save_shard_file, 'w', is_check_file_size=False)
                if safe_serialization:
                    safe_save_file(shard, save_shard_file, metadata={"format": "pt"})
                else:
                    save_function(shard, save_shard_file)
        if self.quantize:
            self.generate_description(save_directory)

    def generate_description(self, save_directory: Optional[str] = None):
        """Generate description file of saved quant model."""
        model_description = {}
        state_dict = unwrap_model_state_dict(self.state_dict())
        quantize_type = self.quantize.upper()
        model_description['model_quant_type'] = quantize_type
        for name, tensor in state_dict.items():
            if '.weight' in name or '.bias' in name:
                if tensor.dtype in QUANTIZE_DTYPE_LIST:
                    model_description[name] = quantize_type
                else:
                    model_description[name] = 'FLOAT'
            else:
                model_description[name] = quantize_type
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            save_directory = os.path.realpath(save_directory)
            save_path = os.path.join(save_directory, f'quant_model_description_{quantize_type.lower()}.json')
            with file_utils.safe_open(save_path, 'w', encoding='utf-8', is_exist_ok=False) as fw:
                json.dump(model_description, fw, indent=4)
        return model_description


@dataclass
class LinearInfo:
    """Dataclass maintaining linear layer information."""
    is_pack: bool = True
    is_all_float: bool = False
    pack_linear: nn.Module = None
    location: str = None
    split_num: int = 1


@dataclass
class AttnLinearInfo(LinearInfo):
    """Dataclass maintaining attention linear layer information."""
    q_linear: nn.Module = None
    k_linear: nn.Module = None
    v_linear: nn.Module = None
    dense_linear: nn.Module = None
    location: str = "BEFORE_ATTN"


@dataclass
class MlpLinearInfo(LinearInfo):
    """Dataclass maintaining MlP linear layer information."""
    up_weight_only: bool = False
    gate_linear: nn.Module = None
    up_linear: nn.Module = None
    down_linear: nn.Module = None
    location: str = "BEFORE_MLP"


@dataclass
class LmHeadLinearInfo(LinearInfo):
    """Dataclass maintaining languange model head linear layer information."""
    lm_head_name: str = None
    lm_head_linear: nn.Module = None
    location: str = "BEFORE_LMHEAD"


def safe_get_tokenizer_from_pretrained(model_path: str, **kwargs) -> AutoTokenizer:
    """A wrapper function of `AutoTokenizer.from_pretrained` which validates the path."""
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)
    except EnvironmentError:
        raise EnvironmentError("Get tokenizer from pretrained failed, "
                               "please check tokenizer files in the model path. "
                               "If the file exists, "
                               "make sure the folder's owner has execute permission.") from None
    except Exception:
        raise ValueError("Get tokenizer from pretrained failed, "
                         "please check the input parameters model_path and kwargs. "
                         "If the file exists, "
                         "make sure the folder's owner has execute permission.") from None
    return tokenizer


def safe_get_model_from_pretrained(model_path: str, **kwargs) -> AutoModelForCausalLM:
    """A wrapper of `AutoModelForCausalLM.from_pretrained` which validates the path."""
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)
    except EnvironmentError:
        raise EnvironmentError("Get model from pretrained failed, "
                               "please check model weights files in the model path. "
                               "If the file exists, "
                               "make sure the folder's owner has execute permission.") from None
    except Exception:
        raise ValueError("Get model from pretrained failed, "
                         "please check the input parameters model_path and kwargs. "
                         "If the file exists, "
                         "make sure the folder's owner has execute permission.") from None
    return model


def safe_get_auto_model_from_pretrained(model_path: str, **kwargs) -> AutoModel:
    """A wrapper of `AutoModel.from_pretrained` which validates the path."""
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        model = AutoModel.from_pretrained(model_path, local_files_only=True, **kwargs)
    except EnvironmentError:
        raise EnvironmentError("Get auto_model from pretrained failed, "
                               "please check model weights files in the model path. "
                               "If the file exists, "
                               "make sure the folder's owner has execute permission.") from None
    except Exception:
        raise ValueError("Get model from pretrained failed, "
                         "please check the input parameters model_path and kwargs. "
                         "If the file exists, "
                         "make sure the folder's owner has execute permission.") from None
    return model


def safe_get_auto_model_for_sequence_classification_from_pretrained(
        model_path:str, **kwargs
    ) -> AutoModelForSequenceClassification:
    """A wrapper of `AutoModelForSequenceClassification.from_pretrained` which validates the path."""
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, **kwargs)
    except EnvironmentError:
        raise EnvironmentError("Get auto_model_for_sequence_classification from pretrained failed, "
                               "please check model weights files in the model path. "
                               "If the file exists, "
                               "make sure the folder's owner has execute permission.") from None
    except Exception:
        raise ValueError("Get model from pretrained failed, "
                         "please check the input parameters model_path and kwargs. "
                         "If the file exists, "
                         "make sure the folder's owner has execute permission.") from None
    return model


def safe_get_config_from_pretrained(model_path: str, **kwargs) -> AutoConfig:
    """A wrapper of `AutoConfig.from_pretrained` which will validate the path."""
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, **kwargs)
    except EnvironmentError:
        raise EnvironmentError("Get config from pretrained failed, please check config file in the model path. "
                               "If the file exists, "
                               "make sure the folder's owner has execute permission.") from None
    except Exception:
        raise ValueError("Get config from pretrained failed, please check config file in the model path. "
                         "If the file exists, "
                         "make sure the folder's owner has execute permission.") from None
    return config


def safe_get_config_dict(model_path: str, **kwargs) -> PretrainedConfig:
    """A wrapper of `PretrainedConfig.get_config_dict` which will validate the path."""
    model_path = file_utils.standardize_path(model_path, check_link=False)
    file_utils.check_path_permission(model_path)
    try:
        config, _ = PretrainedConfig.get_config_dict(model_path, local_files_only=True, **kwargs)
    except EnvironmentError:
        raise EnvironmentError("Get config from pretrained failed, please check config file in the model path. "
                               "If the file exists, "
                               "make sure the folder's owner has execute permission. ") from None
    except Exception:
        raise ValueError("Get config from pretrained failed, please check config file in the model path. "
                         "If the file exists, "
                         "make sure the folder's owner has execute permission.") from None
    return config


def safe_from_pretrained(target_cls: Type, model_path: Optional[str] = None, **kwargs) -> Any:
    """
    A wrapper of `target_cls.from_pretrained` which will validate the path.
    
    Args:
        target_cls (Type): The class to be instantiated.
        model_path (str, optional): The path to the pretrained model, defaults to None.
        **kwargs: Additional keyword arguments.
    
    Returns:
        Any: The instantiated class.
    """
    if model_path:
        model_path = file_utils.standardize_path(model_path, check_link=False)
        file_utils.check_path_permission(model_path)

    try:
        if model_path:
            target_instance = target_cls.from_pretrained(model_path, local_files_only=True, **kwargs)
        else:
            target_instance = target_cls.from_pretrained(**kwargs, local_files_only=True)
    except EnvironmentError:
        raise EnvironmentError(f"Get instance from {target_cls.__name__} pretrained failed. "
                               "Please check config file in the model path or init parameters. "
                               "If the file exists, "
                               "make sure the folder's owner has execute permission.") from None
    except Exception:
        raise ValueError(f"Get instance from {target_cls.__name__} pretrained failed. "
                         "Please check config file in the model path or init parameters. "
                         "If the file exists, "
                         "make sure the folder's owner has execute permission.") from None
    return target_instance


def safe_open_clip_from_pretrained(open_clip_method: Callable, model_name: str, 
                                   model_path: Optional[str] = None, **kwargs) -> Any:
    """A wrapper of `open_clip_method` which will validate the model_name and model_path."""
    if model_path:
        # 对model_path标准化后进行安全校验
        model_path = file_utils.standardize_path(model_path, check_link=False)
        file_utils.check_file_safety(model_path)

    # 对model_name进行检查，防止open_clip创建该模型时存在联网操作
    hf_hub_prefix = 'hf-hub:'
    if model_name.startswith(hf_hub_prefix):
        raise ValueError(f"Model name should not start with {hf_hub_prefix} to avoid internet connection")

    try:
        if model_path:
            # 如果model_path存在，则判断输入的方法为模型创建方法
            return open_clip_method(model_name, pretrained=model_path, **kwargs)
        else:
            # 判断为其他方法
            return open_clip_method(model_name, **kwargs)
    except EnvironmentError:
        raise EnvironmentError(f"Failed to conduct {open_clip_method.__name__} for open clip. "
                               "Please check config file in the model path or init parameters.") from None
    except Exception:
        raise ValueError(f"Failed to conduct {open_clip_method.__name__} for open clip. "
                         "Please check config file in the model path or init parameters.") from None
