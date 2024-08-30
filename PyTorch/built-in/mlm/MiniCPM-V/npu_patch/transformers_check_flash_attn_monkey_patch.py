from typing import Optional, Union, Dict
import torch
import transformers


def check_flash_attn_2(
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
        hard_check_only: bool = False,
):
    return config


def replace_with_torch_npu_check_flash_attn_2():
    transformers.modeling_utils.PreTrainedModel._check_and_enable_flash_attn_2 = check_flash_attn_2
