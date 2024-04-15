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


import sys
import subprocess
import platform
import argparse

import numpy as np
import cv2


def main(args):
    outputs_bin = args.om_pred

    with open(outputs_bin, 'rb') as f:
        pred_data = f.read()

    pred = np.frombuffer(pred_data, dtype=np.float32).reshape(-1, 3, 96, 96)
    pred = pred.transpose(0, 2, 3, 1) * 255

    frames_bin = args.frames
    coords_bin = args.coords
    frames = np.fromfile(frames_bin, dtype=np.uint8).reshape(-1, 480, 720, 3)
    coords = np.fromfile(coords_bin, dtype=np.int64).reshape(-1, 4)

    out = cv2.VideoWriter('temp/result_om.avi', 
                          cv2.VideoWriter_fourcc(*'DIVX'), 25, (720, 480))

    for p, f, c in zip(pred, frames, coords):
        y1, y2, x1, x2 = c
        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

        f_copy = np.empty_like(f, dtype=np.uint8)
        f_copy[:] = f
        f_copy[y1:y2, x1:x2] = p
        out.write(f_copy)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result_om.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--om_pred', type=str, 
					    help="Filepath for the prediction results of the OM model's output", required=True)
    parser.add_argument('--frames', type=str,
                        help="Bin filepath of frames", required=True)
    parser.add_argument('--coords', type=str,
                        help="Bin filepath of coords", required=True)
    parser.add_argument('--audio', type=str,
                        help="audio filepath", required=True)
    parser.add_argument('--outfile', type=str,
                        help="Video path to save result. See default for an e.g.",
                        default='results/result_voice.mp4')

    args = parser.parse_args()
    main(args)
    