import os
import stable_audio_tools


def main():
    stable_audio_tools_path = stable_audio_tools.__path__

    os.system(f'patch -p0 {stable_audio_tools_path[0]}/inference/generation.py precision_generation.patch')


if __name__ == '__main__':
    main()