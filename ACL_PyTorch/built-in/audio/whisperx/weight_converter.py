import os
import argparse

from safetensors.torch import load_file
import torch

replace_map = {
    # conv
    "model.encoder.conv1": "encoder.conv1",
    "model.encoder.conv2": "encoder.conv2",
    # layer norm
    "model.encoder.layer_norm": "encoder.ln_post",
    "model.decoder.layer_norm": "decoder.ln",
    "encoder_attn_layer_norm": "cross_attn_ln",
    # encoder and decoder
    "model.encoder.layers": "encoder.blocks",
    "model.decoder.layers": "decoder.blocks",
    # attention
    "self_attn.q_proj": "attn.query",
    "self_attn.k_proj": "attn.key",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.k_proj": "cross_attn.key",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    # fc -> mlp
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    # embedding
    "model.encoder.embed_positions.weight": "encoder.positional_embedding",
    "model.decoder.embed_positions.weight": "decoder.positional_embedding",
    "model.decoder.embed_tokens": "decoder.token_embedding"
}


def convert_key(old_key):
    new_key = old_key
    for old_part, new_part in replace_map.items():
        new_key = new_key.replace(old_part, new_part)
    return new_key


def convert_safetensors_to_pt(safetensor):
    model_state_dict = {}
    for k in safetensor.keys():
        new_k = convert_key(k)
        model_state_dict[new_k] = safetensor[k]
    return model_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        choices=["large-v3", "large-v3-turbo"],
                        default="large-v3",
                        help="choose model, supported models: large-v3 or large-v3-turbo")
    parser.add_argument("--model_path", type=str, default="./weight/whisper-large-v3")
    parser.add_argument("--save_pt_path", type=str, default="./weight/Whisper-large-v3")
    args = parser.parse_args()

    if args.model_name == "large-v3":
        model_file1 = f"{args.model_path}/model.fp32-00001-of-00002.safetensors"
        model_file2 = f"{args.model_path}/model.fp32-00002-of-00002.safetensors"

        if not os.path.exists(model_file1):
            raise FileNotFoundError(f"model file not found: {model_file1}")
        if not os.path.exists(model_file2):
            raise FileNotFoundError(f"model file not found: {model_file2}")

        whisper_safetensor1 = load_file(model_file1)
        whisper_safetensor2 = load_file(model_file2)
        whisper_safetensor = {**whisper_safetensor1, **whisper_safetensor2}

        dims_info = {
            "n_mels": 128,
            "n_vocab": 51866,
            "n_audio_ctx": 1500,
            "n_audio_state": 1280,
            "n_audio_head": 20,
            "n_audio_layer": 32,
            "n_text_ctx": 448,
            "n_text_state": 1280,
            "n_text_head": 20,
            "n_text_layer": 32
        }
    else:
        model_file = f"{args.model_path}/model.safetensors"

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"model file not found: {model_file}")

        whisper_safetensor = load_file(model_file)

        dims_info = {
            "n_mels": 128,
            "n_vocab": 51866,
            "n_audio_ctx": 1500,
            "n_audio_state": 1280,
            "n_audio_head": 20,
            "n_audio_layer": 32,
            "n_text_ctx": 448,
            "n_text_state": 1280,
            "n_text_head": 20,
            "n_text_layer": 4
        }

    model_state_dict = convert_safetensors_to_pt(whisper_safetensor)
    whisper_pt = {"dims": dims_info,
                  "model_state_dict": model_state_dict}
    os.makedirs(args.save_pt_path, exist_ok=True)
    torch.save(whisper_pt, f"{args.save_pt_path}/{args.model_name}.pt")
