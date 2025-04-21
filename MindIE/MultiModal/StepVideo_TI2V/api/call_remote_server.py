import os
import sys
import ast
import threading
import pickle
import argparse
import torch
import torch_npu
from flask import Flask, Response, jsonify, request, Blueprint
from flask_restful import Api, Resource
import torch.distributed as dist

dtype = torch.bfloat16


def parsed_args():
    parser = argparse.ArgumentParser(description="StepVideo API Functions")
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--clip_dir', type=str, default='hunyuan_clip')
    parser.add_argument('--llm_dir', type=str, default='step_llm')
    parser.add_argument('--vae_dir', type=str, default='vae')
    parser.add_argument('--port', type=str, default='5001')

    parser.add_argument('--enable_vae', type=ast.literal_eval, default=False)
    parser.add_argument('--enable_llm', type=ast.literal_eval, default=False)
    args = parser.parse_args()
    return args


class StepVaePipeline(Resource):
    def __init__(self, vae_dir, version=2, device='npu'):
        self.vae = self.build_vae(vae_dir, version).to(device)
        self.scale_factor = 1.0

    def build_vae(self, vae_dir, version=2):
        from stepvideo.vae.vae import AutoencoderKL
        (model_name, z_channels) = ("vae_v2.safetensors", 64) if version == 2 else ("vae.safetensors", 16)
        model_path = os.path.join(vae_dir, model_name)

        model = AutoencoderKL(
            z_channels=z_channels,
            model_path=model_path,
            version=version,
        ).to(dtype).npu().eval()
        print("Inintialized vae...")
        return model

    def decode(self, samples, *args, **kwargs):
        with torch.no_grad():
            dtype = next(self.vae.parameters()).dtype
            device = next(self.vae.parameters()).device
            samples = self.vae.decode(samples.to(dtype).to(device) / self.scale_factor)
            if hasattr(samples, 'sample'):
                samples = samples.sample
            return samples

    def encode(self, videos, *args, **kwargs):
        with torch.no_grad():
            dtype = next(self.vae.parameters()).dtype
            device = next(self.vae.parameters()).device
            latents = self.vae.encode(videos.to(dtype).to(device)) * self.scale_factor
            if hasattr(latents, 'sample'):
                latents = latents.sample
            return latents


lock = threading.Lock()


class VAEapi(Resource):
    def __init__(self, vae_pipeline):
        self.vae_pipeline = vae_pipeline

    def get(self):
        with lock:
            feature = pickle.loads(request.get_data())
            feature['api'] = 'vae'
            feature = {k: v for k, v in feature.items() if v is not None}
            video_latents = self.vae_pipeline.decode(**feature)
            response = pickle.dumps(video_latents)
            return Response(response)


class VAEEncodeapi(Resource):
    def __init__(self, vae_pipeline):
        self.vae_pipeline = vae_pipeline

    def get(self):
        with lock:
            feature = pickle.loads(request.get_data())
            feature['api'] = 'vae-encode'
        
            feature = {k: v for k, v in feature.items() if v is not None}
            video_latents = self.vae_pipeline.encode(**feature)
            response = pickle.dumps(video_latents)
            
            return Response(response)


class CaptionPipeline(Resource):
    def __init__(self, llm_dir, clip_dir, device='npu'):
        self.text_encoder = self.build_llm(llm_dir).to(device)
        self.clip = self.build_clip(clip_dir).to(device)

    def build_llm(self, model_dir):
        from stepvideo.text_encoder.stepllm import STEP1TextEncoder
        text_encoder = STEP1TextEncoder(model_dir, max_length=320).to(dtype).eval()
        print("Inintialized text encoder...")
        return text_encoder

    def build_clip(self, model_dir):
        from stepvideo.text_encoder.clip import HunyuanClip
        clip = HunyuanClip(model_dir, max_length=77).eval()
        print("Inintialized clip encoder...")
        return clip

    def embedding(self, prompts, *args, **kwargs):
        with torch.no_grad():
            y, y_mask = self.text_encoder(prompts)
            
            clip_embedding, _ = self.clip(prompts)
            
            len_clip = clip_embedding.shape[1]
            # pad attention_mask with clip's length
            y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0), value=1)

            data = {
                'y': y.detach().cpu(),
                'y_mask': y_mask.detach().cpu(),
                'clip_embedding': clip_embedding.to(torch.bfloat16).detach().cpu()
            }
            
            return data

lock = threading.Lock()


class Captionapi(Resource):
    def __init__(self, caption_pipeline):
        self.caption_pipeline = caption_pipeline

    def get(self):
        with lock:
            feature = pickle.loads(request.get_data())
            feature['api'] = 'caption'
            feature = {k: v for k, v in feature.items() if v is not None}
            
            embeddings = self.caption_pipeline.embedding(**feature)
            response = pickle.dumps(embeddings)
            
            return Response(response)


class RemoteServer(object):
    def __init__(self, args) -> None:
        self.enable_vae = args.enable_vae
        self.enable_llm = args.enable_llm
        if not self.enable_vae and not self.enable_llm:
            raise ValueError
        elif self.enable_vae and self.enable_llm:
            print("warning: may out of memory on Ascend*")
        self.app = Flask(__name__)
        root = Blueprint("root", __name__)
        self.app.register_blueprint(root)
        api = Api(self.app)
        import logging
        self.app.debug = True
        self.app.logger.setLevel(logging.DEBUG)

        if self.enable_vae:
            self.vae_pipeline = StepVaePipeline(
                vae_dir=os.path.join(args.model_dir, args.vae_dir)
            )
            api.add_resource(
                VAEapi,
                "/vae-api",
                resource_class_args=[self.vae_pipeline],
            )

            api.add_resource(
                VAEEncodeapi,
                "/vae-encode-api",
                resource_class_args=[self.vae_pipeline],
            )

        if self.enable_llm:
            self.caption_pipeline = CaptionPipeline(
                llm_dir=os.path.join(args.model_dir, args.llm_dir),
                clip_dir=os.path.join(args.model_dir, args.clip_dir)
            )
            api.add_resource(
                Captionapi,
                "/caption-api",
                resource_class_args=[self.caption_pipeline],
            )

    def run(self, host="0.0.0.0", port=8080):
        if self.enable_vae:
            port = 5000
            print(f"enable vae, port setting to {port}")
        if self.enable_llm:
            port = 5001
            print(f"enable llm, port setting to {port}")
        self.app.run(host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    args = parsed_args()
    flask_server = RemoteServer(args)
    flask_server.run(host="0.0.0.0", port=args.port)
    torch_npu.npu.config.allow_internal_format = False