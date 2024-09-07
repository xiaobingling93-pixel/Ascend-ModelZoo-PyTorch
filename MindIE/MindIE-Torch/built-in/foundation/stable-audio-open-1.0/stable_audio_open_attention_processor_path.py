import os
import diffusers


def main():
    diffusers_path = diffusers.__path__
    diffusers_version = diffusers.__version__

    assert diffusers_version is not '0.30.0', "expectation diffusers==0.30.0"
    os.system(f'patch -p0 {diffusers_path[0]}/models/attention_processor.py attention_processor.patch')


if __name__ == '__main__':
    main()