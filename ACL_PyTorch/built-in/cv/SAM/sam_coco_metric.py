# Copyright 2025 Huawei Technologies Co., Ltd
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
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from ais_bench.infer.interface import InferSession

from sam_preprocessing_pytorch import encoder_preprocessing, decoder_preprocessing
from sam_postprocessing_pytorch import sam_postprocessing


def rle_to_mask(rle, h, w):
    """COCO segmentation → binary mask (h,w) uint8."""
    if isinstance(rle, list):
        rles = maskUtils.frPyObjects(rle, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(rle, dict) and isinstance(rle.get("counts"), list):
        rle = maskUtils.frPyObjects(rle, h, w)
    return maskUtils.decode(rle).astype(np.uint8)


def compute_iou(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def coco_bbox_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def encoder_infer(session_encoder, x):
    encoder_outputs = session_encoder.infer([x])
    image_embedding = encoder_outputs[0]
    return image_embedding


def decoder_infer(session_decoder, decoder_inputs):
    decoder_outputs = session_decoder.infer(decoder_inputs, mode="dymdims", custom_sizes=[1000, 1000000])
    low_res_masks = decoder_outputs[1]
    return low_res_masks


def save_mask_overlay(masks, image, save_dir, image_name):
    overlay = image.copy()
    alpha = 0.5

    for mask in masks:
        if mask.sum() == 0:
            continue
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # 每个实例随机颜色
        overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + color * alpha).astype(np.uint8)

    base, ext = os.path.splitext(image_name)
    save_path = os.path.join(save_dir, f"{base}_sam_pre{ext}")
    cv2.imwrite(save_path, overlay)


def evaluate_sam_on_coco(coco_root, save_path, encoder, decoder, max_instances=0):
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    img_root = os.path.join(coco_root, "val2017")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO annotations not found: {ann_file}")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"COCO val2017 images not found: {img_root}")

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    session_encoder = encoder
    session_decoder = decoder
    
    ious = []
    counted = 0

    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_root, img_info["file_name"])
        image = cv2.imread(img_path)

        H, W = image.shape[:2]

        x = encoder_preprocessing(image)
        image_embedding = encoder_infer(session_encoder, x)

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        
        mask_list = []
        for ann in anns:
            
            if max_instances > 0 and counted >= max_instances:
                break
            
            box_xyxy = coco_bbox_to_xyxy(ann["bbox"])
            
            decoder_inputs = decoder_preprocessing(image_embedding, box=box_xyxy, image=image)
            low_res_masks = decoder_infer(session_decoder, decoder_inputs)
            masks = sam_postprocessing(low_res_masks, image)

            pred2d = masks[0][0].astype(np.uint8)
            mask_list.append(pred2d)
            pred_bin = pred2d.astype(np.uint8)

            gt_mask = rle_to_mask(ann["segmentation"], H, W)
            iou = compute_iou(pred_bin, gt_mask)
            ious.append(iou)
            counted += 1
            
        if save_path is not None and len(mask_list) > 0:
            save_mask_overlay(mask_list, image, save_path, img_info["file_name"])

        if max_instances > 0 and counted >= max_instances:
            break

    miou = float(np.mean(ious)) if counted > 0 else 0.0
    print("\n=========== COCO Evaluation (Box Prompt) ===========")
    print(f"Instances Evaluated : {counted}")
    print(f"Mean IoU (mIoU)     : {miou:.4f}")
    print("====================================================\n")


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [int(v) for v in value.split(',')]
        for ivalue in ilist[:2]:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError("{} of device:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, value, min_value, max_value))
        return ilist[:2]
    else:
        # default as single int value
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError("device:{} is invalid. valid value range is [{}, {}]".format(
                ivalue, min_value, max_value))
        return ivalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='./datasets/', help='input path to coco dataset')
    parser.add_argument('--save-path', type=str, default=None, help='output path to image')
    parser.add_argument('--encoder-model-path', type=str, default='./models/encoder_sim.om', help='path to encoder model')
    parser.add_argument('--decoder-model-path', type=str, default='./models/decoder_sim.om', help='path to decoder model')
    parser.add_argument('--device-id', type=check_device_range_valid, default=0, help='NPU device id.')
    parser.add_argument('--max-instances', type=int, default=0, help='Maximum number of instances to evaluate (0 = all).')
    args = parser.parse_args()

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(os.path.realpath(args.save_path), mode=0o744)

    session_encoder = InferSession(args.device_id, args.encoder_model_path)
    session_decoder = InferSession(args.device_id, args.decoder_model_path)

    evaluate_sam_on_coco(
        args.dataset_path,
        args.save_path,
        session_encoder,
        session_decoder,
        max_instances=args.max_instances
    )

if __name__ == "__main__":
    main()

