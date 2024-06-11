# Copyright 2024 Huawei Technologies Co., Ltd
import types

import torch
from torch import nn
from transformers import T5EncoderModel, CLIPModel, CLIPProcessor
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Attention
from transformers.activations import NewGELUActivation

from opensora.utils.utils import get_precision
from opensora.utils.npu_utils import is_npu_available, NpuRMSNorm, replace_module, t5_forward


class T5Wrapper(nn.Module):
    def __init__(self, args):
        super(T5Wrapper, self).__init__()
        self.model_name = args.text_encoder_name
        dtype = get_precision(args)
        t5_model_kwargs = {'cache_dir': './cache_dir', 'low_cpu_mem_usage': True, 'torch_dtype': dtype}
        self.text_enc = T5EncoderModel.from_pretrained(self.model_name, **t5_model_kwargs).eval()
        if is_npu_available():
            # Monkey Patch NpuRMSNorm, T5FA and GELU
            for name, module in self.text_enc.named_modules():
                if isinstance(module, T5Attention):
                    module.forward = types.MethodType(t5_forward, module)
                if isinstance(module, T5LayerNorm):
                    hidden_size = module.weight.shape[0]
                    eps = module.variance_epsilon
                    npu_rms_norm = NpuRMSNorm(hidden_size, eps)
                    npu_rms_norm.load_state_dict(module.state_dict())
                    replace_module(self.text_enc, name, npu_rms_norm)
                if isinstance(module, NewGELUActivation):
                    replace_module(self.text_enc, name, nn.GELU(approximate='tanh'))

    def forward(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()

class CLIPWrapper(nn.Module):
    def __init__(self, args):
        super(CLIPWrapper, self).__init__()
        self.model_name = args.text_encoder_name
        dtype = get_precision(args)
        model_kwargs = {'cache_dir': './cache_dir', 'low_cpu_mem_usage': True, 'torch_dtype': dtype}
        self.text_enc = CLIPModel.from_pretrained(self.model_name, **model_kwargs).eval()

    def forward(self, input_ids, attention_mask): 
        text_encoder_embs = self.text_enc.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return text_encoder_embs.detach()



text_encoder = {
    'DeepFloyd/t5-v1_1-xxl': T5Wrapper,
    'openai/clip-vit-large-patch14': CLIPWrapper
}


def get_text_enc(args):
    """deprecation"""
    text_enc = text_encoder.get(args.text_encoder_name, None)
    assert text_enc is not None
    return text_enc(args)
