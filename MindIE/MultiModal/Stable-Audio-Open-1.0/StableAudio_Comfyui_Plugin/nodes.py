import os
import json
import torch
import torch_npu
from safetensors.torch import load_file

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import comfy.supported_models_base
import comfy.sd
from comfy import utils
import folder_paths
from folder_paths import get_folder_paths, get_filename_list_

from diffusers.models.embeddings import get_1d_rotary_pos_embed
from mindiesd import CacheConfig, CacheAgent

from .stableaudio import (
    StableAudioPipeline,
    StableAudioDiTModel,
    AutoencoderOobleck,
)

if "mindiesd" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["mindiesd"][0].append(
        os.path.join(folder_paths.models_dir, "mindiesd"))
    folder_paths.folder_names_and_paths["mindiesd"][1].add(".pt").add(".ts").add(".bin").add(".safetensors")
else:
    folder_paths.folder_names_and_paths["mindiesd"] = (
        [os.path.join(folder_paths.models_dir, "mindiesd")], {".pt", ".ts", ".bin", ".safetensors"}
    )

MODEL_FOLDER_PATH = folder_paths.get_folder_paths("mindiesd")[0]
PIPELINE_FOLDER_PATH = os.path.join(MODEL_FOLDER_PATH, "stable-audio-open-1.0")
UNET_FOLDER_PATH = os.path.join(PIPELINE_FOLDER_PATH, "transformer")


class UnetAdapter(torch.nn.Module):
    def __init__(self, engine_path, use_ditcache, use_attentioncache, start_step, attentioncache_interval, end_step):
        super().__init__()
        if (use_ditcache and use_attentioncache):
            raise ValueError(f"Only support one cache at a time, but got use_ditcache is {use_ditcache}, and use_attentioncache is {use_attentioncache}")
        self.device = comfy.model_management.get_torch_device()
        self.dtype = torch.float16
        self.unet_model = StableAudioDiTModel.from_pretrained(os.path.dirname(engine_path), 
                        torch_dtype=self.dtype)
        transformer = self.unet_model
        self.use_ditcache = use_ditcache
        self.step = 0
        if use_attentioncache:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=len(transformer.transformer_blocks),
                steps_count=100,
                step_start=start_step,
                step_interval=attentioncache_interval,
                step_end=end_step
            )
        else:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=len(transformer.transformer_blocks),
                steps_count=100
            )
        cache = CacheAgent(config)
        for block in transformer.transformer_blocks:
            block.cache = cache
        
        self.unet_model = self.unet_model.to(self.device).eval()
        self.unet_model.config.addition_embed_type = None

        # freqs
        rotary_embedding = get_1d_rotary_pos_embed(
            32,
            1025,
            use_real=True,
            repeat_interleave_real=False,
        )
        cos = rotary_embedding[0][None, :, None, :].to(self.device).to(self.dtype)
        sin = rotary_embedding[1][None, :, None, :].to(self.device).to(self.dtype)
        self.rotary_embedding = (cos, sin)

    def __call__(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        global_embed = kwargs.get("global_embed", None)
        noise_pred = self.unet_model(
            self.step,
            x,
            timesteps,
            context,
            global_embed.unsqueeze(1),
            rotary_embedding=self.rotary_embedding,
            return_dict=False,
            use_cache=self.use_ditcache,
        )[0]
        self.step = (self.step + 1) % 100
        return noise_pred


class StableAudioUnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("mindiesd"), {"tooltip": "The name of the transformer (model) to load."}),
                             "use_ditcache": ("BOOLEAN", {"default": True, "tooltip": "Use DiT Cache."}),
                             "use_attentioncache": ("BOOLEAN", {"default": False, "tooltip": "Use AGB Cache."}),
                             "start_step": ("INT", {"default": 60, "min": 10, "max": 10000, "tooltip": ""}),
                             "attentioncache_interval": ("INT", {"default": 5, "min": 1, "max": 10, "tooltip": ""}),
                             "end_step": ("INT", {"default": 97, "min": 1, "max": 10000, "tooltip": ""}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "MindIE SD/StableAudio"

    def load_unet(self, unet_name, use_ditcache, use_attentioncache, start_step, attentioncache_interval, end_step):
        
        unet_path = os.path.join(MODEL_FOLDER_PATH, unet_name)
        unet = UnetAdapter(unet_path, use_ditcache, use_attentioncache, start_step, attentioncache_interval, end_step)
        conf = comfy.supported_models.StableAudio({"audio_model": "dit1.0"})
        conf.unet_config["disable_unet_model_creation"] = True
        conf.unet_config = {
                "audio_model": "dit1.0",
        }

        project_path = os.path.join("projection_model", "diffusion_pytorch_model.safetensors")
        project_model = load_file(os.path.join(PIPELINE_FOLDER_PATH, project_path))
        seconds_start_embedder_weights = utils.state_dict_prefix_replace(project_model, {"start_number_conditioner.time_positional_": "embedder."}, filter_keys=True)
        seconds_total_embedder_weights = utils.state_dict_prefix_replace(project_model, {"end_number_conditioner.time_positional_": "embedder."}, filter_keys=True)

        model = comfy.model_base.StableAudio1(conf, seconds_start_embedder_weights, seconds_total_embedder_weights)

        model.diffusion_model = unet
        model.memory_required = lambda *args, **kwargs: 0 #always pass inputs batched up as much as possible, our TRT code will handle batch splitting

        return (comfy.model_patcher.ModelPatcher(model,
                                                 load_device=comfy.model_management.get_torch_device(),
                                                 offload_device=comfy.model_management.unet_offload_device()),)

NODE_CLASS_MAPPINGS = {
"StableAudioUnetLoader": StableAudioUnetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
"StableAudioUnetLoader": "MindIE SD StableAudio Unet Loader",
}
