from setuptools import find_packages, setup
import subprocess

if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("stepvideo/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="stepvideo",
        author="Step-Video Team",
        packages=find_packages(),
        install_requires=[
            "torchvision==0.16.0",
            "torch==2.1.0",
            "accelerate>=1.0.0",
            "transformers>=4.39.1",
            "diffusers>=0.31.0",
            "sentencepiece>=0.1.99",
            "imageio>=2.37.0",
            "imageio-ffmpeg",
            "optimus==2.1",
            "numpy",
            "ninja",
            "einops",
            "aiohttp",
            "flask",
            "flask_restful",
            "ffmpeg-python",
            "requests",
            "yunchang==0.6.0",
        ],
        url="",
        description="A 30B DiT based text to video and image generation model",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )