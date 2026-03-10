import argparse
import time
import warnings

from deoldify.visualize import get_image_colorizer, get_video_colorizer

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu


def infer_image(art, image_path, image_url, image_render_factor, watermarked, warm_num, infer_num):
    colorizer = get_image_colorizer(artistic=art)
    render_factor = image_render_factor
    source_path = image_path
    source_url = image_url

    # warm up
    print(f"start warm up")
    for _ in range(warm_num):
        if source_url is not None:
            colorizer.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=False, watermarked=watermarked)
        else:
            colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=False, watermarked=watermarked)
    
    print(f"start image colorize")
    if source_url is not None:
        print(f"image_name: {source_url}")
    else:
        print(f"image_name: {source_path}")
    for i in range(infer_num):
        print(f"#################### {i} ####################")
        st = time.time()
        if source_url is not None:
            result_path = colorizer.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=False, watermarked=watermarked)
        else:
            result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=False, watermarked=watermarked)
        print(f"image infer cost {time.time() - st} /s")
    print(f"result image path: {result_path}")


def infer_video(video_path, video_url, video_render_factor, watermarked, warm_num, infer_num):
    colorizer = get_video_colorizer()
    render_factor = video_render_factor
    source_path = video_path
    source_url = video_url

    # warm up
    print(f"start warm up")
    for _ in range(warm_num):
        if source_url is not None:
            colorizer.colorize_from_url(source_url, source_path, render_factor=render_factor, watermarked=watermarked)
        else:
            colorizer.colorize_from_file_name(source_path, render_factor=render_factor, watermarked=watermarked)
    
    print(f"start video colorize")
    if source_url is not None:
        print(f"video_name: {source_url}")
    else:
        print(f"video_name: {source_path}")
    for i in range(infer_num):
        print(f"#################### {i} ####################")
        st = time.time()
        if source_url is not None:
            result_path = colorizer.colorize_from_url(source_url, source_path, render_factor=render_factor, watermarked=watermarked)
        else:
            result_path = colorizer.colorize_from_file_name(source_path, render_factor=render_factor, watermarked=watermarked)
        print(f"video infer cost {time.time() - st} /s")
    print(f"result video path: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-art", "--image_artistic", action="store_true", help="Colorize image with artistic model weight")
    parser.add_argument("-img", "--image", action="store_true", help="Colorize image")
    parser.add_argument("--video", action="store_true", help="Colorize video")
    parser.add_argument("--image_path", type=str, default="test_image.jpg", help="Colorize image path")
    parser.add_argument("--video_path", type=str, default="test_video.mp4", help="Colorize video path")
    parser.add_argument("--image_url", type=str, default=None, help="Colorize image url")
    parser.add_argument("--video_url", type=str, default=None, help="Colorize video url")
    parser.add_argument("--image_render_factor", type=int, default=35, help="Colorize image render_factor")
    parser.add_argument("--video_render_factor", type=int, default=21, help="Colorize video render_factor")
    parser.add_argument("--watermarked", action="store_true", help="use watermarked")
    parser.add_argument("--warm_num", type=int, default=2, help="warm up times")
    parser.add_argument("--infer_num", type=int, default=1, help="infer times")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
    args = parse_args()
    if args.image:
        infer_image(args.image_artistic, args.image_path, args.image_url, args.image_render_factor, args.watermarked, args.warm_num, args.infer_num)
    if args.video:
        infer_video(args.video_path, args.video_url, args.video_render_factor, args.watermarked, args.warm_num, args.infer_num)