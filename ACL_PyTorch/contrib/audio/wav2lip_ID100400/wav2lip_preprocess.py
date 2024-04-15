# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import argparse
import subprocess

import cv2
import torch
import numpy as np
from tqdm import tqdm

import audio
import face_detection

device = 'cpu'


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def gen_img(args):
    if not os.path.isfile(args.video):
        raise ValueError('--video argument must be a valid path to video/image file')

    elif args.video.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.video)]

    else:
        video_stream = cv2.VideoCapture(args.video)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

        return full_frames

def gen_audio(args):
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_step_size = 16
    mel_chunks = []
    mel_idx_multiplier = 80.0 / args.fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    return mel_chunks

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i: i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def gen_data(frames, mels, args):
	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i % len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch = [face]
		mel_batch = [m]
		frame_batch = [frame_to_save]
		coords_batch = [coords]
	
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wav2lip model data preprocess')

    parser.add_argument('--video', type=str, 
					    help='Filepath of video/image that contains faces to use', required=True)

    parser.add_argument('--data_save_dir', type=str, 
                        help='Filepath of video/image that transfers bin file to save', required=True)

    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					    default=25., required=False)

    parser.add_argument('--resize_factor', default=1, type=int, 
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                        'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--audio', type=str, 
                        help='Filepath of video/audio file to use as raw audio source', required=True)

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--face_det_batch_size', type=int, 
                        help='Batch size for face detection', default=16)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    parser.add_argument('--wav2lip_batch_size', type=int, 
                        help='Batch size for Wav2Lip model(s)', default=128)

    args = parser.parse_args()
    args.static = False
    args.img_size = 96
   
    full_frames = gen_img(args)
    mel_chunks = gen_audio(args)
    full_frames = full_frames[:len(mel_chunks)]
    gen = gen_data(full_frames.copy(), mel_chunks, args)

    batches = [(img_batch, mel_batch, frame_batch, coords_batch) for img_batch, mel_batch, frame_batch, coords_batch in gen]  

    # 分别提取每个类型的批次，并进行拼接  
    imgs = np.concatenate([img for img, _, _, _ in batches], axis=0)  
    mels = np.concatenate([mel for _, mel, _, _ in batches], axis=0)  
    frames = np.concatenate([frame for _, _, frame, _ in batches], axis=0)  
    coords = np.concatenate([coord for _, _, _, coord in batches], axis=0)  
    
    imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32)
    mels = np.transpose(mels, (0, 3, 1, 2)).astype(np.float32)

    # 检查文件夹是否存在，如果不存在，则创建它
    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir, mode=0o777, exist_ok=False)
    imgs.tofile(os.path.join(args.data_save_dir, "imgs.bin"))
    mels.tofile(os.path.join(args.data_save_dir, "mels.bin"))
    frames.tofile(os.path.join(args.data_save_dir, "frames.bin"))
    coords.tofile(os.path.join(args.data_save_dir, "coords.bin"))