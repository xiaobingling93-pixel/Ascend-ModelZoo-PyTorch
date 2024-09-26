import os
import diffusers


def main():
    diffusers_path = diffusers.__path__
    diffusers_version = diffusers.__version__

    assert diffusers_version is not '0.30.0', "expectation diffusers==0.30.0"
    os.system(f'patch -p0 {diffusers_path[0]}/models/attention_processor.py attention_processor.patch')
    os.system(f'patch -p0 {diffusers_path[0]}/models/transformers/stable_audio_transformer.py stable_audio_transformer.patch')
    os.system(f'patch -p0 {diffusers_path[0]}/pipelines/stable_audio/pipeline_stable_audio.py pipeline_stable_audio.patch')

if __name__ == '__main__':
    main()