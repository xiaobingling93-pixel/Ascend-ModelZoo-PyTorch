# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import itertools
import json
import os
import re
from collections import OrderedDict
from functools import wraps
from typing import Any, List, Optional, Tuple, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args
from torch import Tensor, nn

from diffusers import __version__
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.quantizers.quantization_config import QuantizationMethod
from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
    deprecate,
    is_accelerate_available,
    is_bitsandbytes_version,
    logging,
)
from diffusers.utils.hub_utils import PushToHubMixin
from diffusers.models.model_loading_utils import (
    _fetch_index_file,
    _fetch_index_file_legacy,
    _load_state_dict_into_model,
    _merge_sharded_checkpoints,
    load_model_dict_into_meta,
    load_state_dict,
)


logger = logging.get_logger(__name__)


_LOW_CPU_MEM_USAGE_DEFAULT = True


if is_accelerate_available():
    import accelerate


def get_parameter_device(parameter: torch.nn.Module) -> torch.device:
    try:
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for param in parameter.parameters():
        last_dtype = param.dtype
        if param.is_floating_point():
            return param.dtype

    for buffer in parameter.buffers():
        last_dtype = buffer.dtype
        if buffer.is_floating_point():
            return buffer.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for current_tuple in gen:
        last_tuple = current_tuple 
        if current_tuple[1].is_floating_point():
            return current_tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype


class ModelMixin(torch.nn.Module, PushToHubMixin):
    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = None
    _no_split_modules = None
    _keep_in_fp32_modules = None

    def __init__(self):
        super().__init__()

    def __getattr__(self, name: str) -> Any:

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'unet.config.{name}'."
            deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False, stacklevel=3)
            return self._internal_dict[name]

        return super().__getattr__(name)

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        quantization_config = kwargs.pop("quantization_config", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set"
                " `device_map=None`. You can install accelerate with `pip install accelerate`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError as e:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                ) from e
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )
        # no in-place modification of the original config.
        config = copy.deepcopy(config)

        # determine initial quantization config.
        #######################################
        pre_quantized = "quantization_config" in config and config["quantization_config"] is not None
        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config["quantization_config"] = DiffusersAutoQuantizer.merge_quantization_configs(
                    config["quantization_config"], quantization_config
                )
            else:
                config["quantization_config"] = quantization_config
            hf_quantizer = DiffusersAutoQuantizer.from_config(
                config["quantization_config"], pre_quantized=pre_quantized
            )
        else:
            hf_quantizer = None

        if hf_quantizer is not None:
            is_bnb_quantization_method = hf_quantizer.quantization_config.quant_method.value == "bitsandbytes"
            if is_bnb_quantization_method and device_map is not None:
                raise NotImplementedError(
                    "Currently, `device_map` is automatically inferred for quantized bitsandbytes models. Support for providing `device_map` as an input will be added in the future."
                )

            hf_quantizer.validate_environment(torch_dtype=torch_dtype, from_flax=from_flax, device_map=device_map)
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `hf_quantizer` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
        )
        if use_keep_in_fp32_modules:
            keep_in_fp32_modules = cls._keep_in_fp32_modules
            if not isinstance(keep_in_fp32_modules, list):
                keep_in_fp32_modules = [keep_in_fp32_modules]

            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.info("Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None.")
            elif not low_cpu_mem_usage:
                raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
        else:
            keep_in_fp32_modules = []
        #######################################

        # Determine if we're loading from a directory of sharded checkpoints.
        is_sharded = False
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file_kwargs = {
            "is_local": is_local,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "subfolder": subfolder or "",
            "use_safetensors": use_safetensors,
            "cache_dir": cache_dir,
            "variant": variant,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "user_agent": user_agent,
            "commit_hash": commit_hash,
        }
        index_file = _fetch_index_file(**index_file_kwargs)
        # In case the index file was not found we still have to consider the legacy format.
        # this becomes applicable when the variant is not None.
        if variant is not None and (index_file is None or not os.path.exists(index_file)):
            index_file = _fetch_index_file_legacy(**index_file_kwargs)
        if index_file is not None and index_file.is_file():
            is_sharded = True

        if is_sharded and from_flax:
            raise ValueError("Loading of sharded checkpoints is not supported when `from_flax=True`.")

        # load model
        model_file = None
        if from_flax:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=FLAX_WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            model = cls.from_config(config, **unused_kwargs)

            # Convert the weights
            from .modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model

            model = load_flax_checkpoint_in_pytorch_model(model, model_file)
        else:
            if is_sharded:
                sharded_ckpt_cached_folder, sharded_metadata = _get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    index_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder or "",
                )
                if hf_quantizer is not None and is_bnb_quantization_method:
                    model_file = _merge_sharded_checkpoints(sharded_ckpt_cached_folder, sharded_metadata)
                    logger.info("Merged sharded checkpoints as `hf_quantizer` is not None.")
                    is_sharded = False

            elif use_safetensors and not is_sharded:
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )

                except IOError as e:
                    logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                    if not allow_pickle:
                        raise
                    logger.warning(
                        "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                    )

            if model_file is None and not is_sharded:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )

            if low_cpu_mem_usage:
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **unused_kwargs)

                if hf_quantizer is not None:
                    hf_quantizer.preprocess_model(
                        model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
                    )

                # if device_map is None, load the state dict and move the params from meta device to the cpu
                if device_map is None and not is_sharded:
                    # `torch.cuda.current_device()` is fine here when `hf_quantizer` is not None.
                    # It would error out during the `validate_environment()` call above in the absence of cuda.
                    if hf_quantizer is None:
                        param_device = "cpu"
                    else:
                        param_device = torch.device(torch.cuda.current_device())
                    state_dict = load_state_dict(model_file, variant=variant)
                    model._convert_deprecated_attention_blocks(state_dict)

                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if hf_quantizer is not None:
                        missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix="")
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_name_or_path,
                        hf_quantizer=hf_quantizer,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        logger.warning(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )

                else:
                    weights_path = index_file
                    with open(index_file) as f:
                        index = json.loads(f.read())
                    if "weight_map" in index:
                        index = index["weight_map"]
                    weights_path = sorted(list(set(index.values())))
                    weights_path = [os.path.join(pretrained_model_name_or_path, f) for f in weights_path]

                    model = cls._load_model(model, weights_path, is_sharded)

                loading_info = {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            else:
                model = cls.from_config(config, **unused_kwargs)

                state_dict = load_state_dict(model_file, variant=variant)
                model._convert_deprecated_attention_blocks(state_dict)

                model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                    model,
                    state_dict,
                    model_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )

                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        # When using `use_keep_in_fp32_modules` if we do a global `to()` here, then we will
        # completely lose the effectivity of `use_keep_in_fp32_modules`.
        elif torch_dtype is not None and hf_quantizer is None and not use_keep_in_fp32_modules:
            model = model.to(torch_dtype)

        if hf_quantizer is not None:
            # We also make sure to purge `_pre_quantization_dtype` when we serialize
            # the model config because `_pre_quantization_dtype` is `torch.dtype`, not JSON serializable.
            model.register_to_config(_name_or_path=pretrained_model_name_or_path, _pre_quantization_dtype=torch_dtype)
        else:
            model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model

    @classmethod
    def _load_model(cls, model, weights_path, is_sharded):
        if not is_sharded:
            state_dict = load_state_dict(weights_path)
            model.load_weights(state_dict)
        else:
            need_key = set(model.state_dict().keys())
            state_dict = {}
            cache = {}
            for weight_file in weights_path:
                state_dict = load_state_dict(weight_file)
                state_dict.update(cache)
                loadkey_cache = model.load_weights(state_dict, is_sharded)
                if loadkey_cache :
                    if isinstance(loadkey_cache, tuple):
                        loaded_keys, cache = loadkey_cache
                    else:
                        loaded_keys = loadkey_cache
                need_key = need_key.symmetric_difference(set(loaded_keys))
                
            if len(need_key) > 0:
                raise ValueError(f"The weight miss key: {need_key}")
        return model

    def load_weights(self, state_dict, shard=False):
        with torch.no_grad():
            if not shard:
                self.load_state_dict(state_dict)
                return {}
            else:
                self.load_state_dict(state_dict, strict=False, assign=True)
                return state_dict.keys()

    # Adapted from `transformers`.
    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "Calling `cuda()` is not supported for `8-bit` quantized models. "
                    " Please use the model as it is, since the model has already been set to the correct devices."
                )
            elif is_bitsandbytes_version("<", "0.43.2"):
                raise ValueError(
                    "Calling `cuda()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        return super().cuda(*args, **kwargs)

    # Adapted from `transformers`.
    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "is_quantized", False):
            if dtype_present_in_args:
                raise ValueError(
                    "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please "
                    "use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`"
                )

        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "`.to` is not supported for `8-bit` bitsandbytes models. Please use the model as it is, since the"
                    " model has already been set to the correct devices and casted to the correct `dtype`."
                )
            elif is_bitsandbytes_version("<", "0.43.2"):
                raise ValueError(
                    "Calling `to()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        return super().to(*args, **kwargs)

    # Taken from `transformers`.
    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been cast to the correct `dtype`."
            )
        else:
            return super().half(*args)

    # Taken from `transformers`.
    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been cast to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict: OrderedDict,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ignore_mismatched_sizes: bool = False,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = list(state_dict.keys())

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @property
    def device(self) -> torch.device:
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)

    def _convert_deprecated_attention_blocks(self, state_dict: OrderedDict) -> None:
        deprecated_attention_block_paths = []

        def recursive_find_attn_block(name, module):
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_paths.append(name)

            for sub_name, sub_module in module.named_children():
                sub_name = sub_name if name == "" else f"{name}.{sub_name}"
                recursive_find_attn_block(sub_name, sub_module)

        recursive_find_attn_block("", self)

        for path in deprecated_attention_block_paths:
            # group_norm path stays the same

            # query -> to_q
            if f"{path}.query.weight" in state_dict:
                state_dict[f"{path}.to_q.weight"] = state_dict.pop(f"{path}.query.weight")
            if f"{path}.query.bias" in state_dict:
                state_dict[f"{path}.to_q.bias"] = state_dict.pop(f"{path}.query.bias")

            # key -> to_k
            if f"{path}.key.weight" in state_dict:
                state_dict[f"{path}.to_k.weight"] = state_dict.pop(f"{path}.key.weight")
            if f"{path}.key.bias" in state_dict:
                state_dict[f"{path}.to_k.bias"] = state_dict.pop(f"{path}.key.bias")

            # value -> to_v
            if f"{path}.value.weight" in state_dict:
                state_dict[f"{path}.to_v.weight"] = state_dict.pop(f"{path}.value.weight")
            if f"{path}.value.bias" in state_dict:
                state_dict[f"{path}.to_v.bias"] = state_dict.pop(f"{path}.value.bias")

            # proj_attn -> to_out.0
            if f"{path}.proj_attn.weight" in state_dict:
                state_dict[f"{path}.to_out.0.weight"] = state_dict.pop(f"{path}.proj_attn.weight")
            if f"{path}.proj_attn.bias" in state_dict:
                state_dict[f"{path}.to_out.0.bias"] = state_dict.pop(f"{path}.proj_attn.bias")
