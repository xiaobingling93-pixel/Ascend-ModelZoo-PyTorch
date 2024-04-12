# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import sys

import cv2

from vision.ssd.mobilenetv1_ssd import (create_mobilenetv1_ssd,
                                        create_mobilenetv1_ssd_predictor)
from vision.utils.misc import Timer

if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <model path> <label path> <image path>')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]
image_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)

predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(
        orig_image,
        label,
        (int(box[0]) + 20, int(box[1]) + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # font scale
        (255, 0, 255),
        2,  # line type
    )
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
