# Copyright 2026 Huawei Technologies Co., Ltd
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
from PIL import Image
from ais_bench.infer.interface import InferSession

from sam2_preprocessing import encoder_preprocessing, decoder_preprocessing
from sam2_postprocessing import decoder_postprocessing


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


def encoder_infer_batch(session_encoder, input_images_batch):
    """
    Batch encoder inference
    input_images_batch: np.array (BS, 3, H, W)
    returns: high_res_feats_0, high_res_feats_1, image_embed (all batched)
    """
    encoder_outputs = session_encoder.infer([input_images_batch])
    high_res_feats_0 = encoder_outputs[0]
    high_res_feats_1 = encoder_outputs[1]
    image_embed = encoder_outputs[2]
    return high_res_feats_0, high_res_feats_1, image_embed


def decoder_infer_single(session_decoder, decoder_inputs):
    """
    Single instance decoder inference (batch=1)
    decoder_inputs: list of 7 np.arrays, each with shape (1, ...)
    returns: masks, iou_predictions, low_res_masks
    """
    decoder_outputs = session_decoder.infer(decoder_inputs)
    masks = decoder_outputs[0]
    iou_predictions = decoder_outputs[1]
    low_res_masks = decoder_outputs[2]
    return masks, iou_predictions, low_res_masks


def save_mask_overlay(masks, image, save_dir, image_name):
    overlay = image.copy()
    alpha = 0.5
    for mask in masks:
        if mask.sum() == 0:
            continue
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + color * alpha).astype(np.uint8)
    base, ext = os.path.splitext(image_name)
    save_path = os.path.join(save_dir, f"{base}_sam_pre{ext}")
    cv2.imwrite(save_path, overlay)


def evaluate_sam_on_coco(coco_root, save_path, session_encoder, session_decoder, 
                          encoder_batch_size=1, max_instances=0):
    """
    COCO evaluation with Encoder batch support, Decoder single instance
    - Encoder: batch 处理多张图片
    - Decoder: 单 instance 处理
    - 流式处理：不累积所有图片，处理完立即释放
    """
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    img_root = os.path.join(coco_root, "val2017")
    
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO annotations not found: {ann_file}")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"COCO val2017 images not found: {img_root}")
    
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    ious = []
    counted = 0
    
    # ========== Batch Buffer (固定大小，流式处理) ==========
    batch_images = []
    
    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_root, img_info["file_name"])
        
        with Image.open(img_path) as image:
            image = np.array(image.convert("RGB"))
            image_orig_hw = image.shape[:2]

            input_image = encoder_preprocessing(image)
        
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            
            batch_images.append({
                'img_info': img_info,
                'image_orig': image,
                'image_orig_hw': image_orig_hw,
                'image_preprocessed': input_image,
                'anns': anns,
            })
            
        if len(batch_images) >= encoder_batch_size:
            counted, batch_ious = _process_batch(
                batch_images, 
                session_encoder, session_decoder,
                save_path,
                max_instances, counted
            )
            ious.extend(batch_ious)
            batch_images = []
        
        if max_instances > 0 and counted >= max_instances:
            break
    
    # ========== 处理剩余不足一个 batch 的数据 ==========
    if len(batch_images) > 0:
        counted, batch_ious = _process_batch(
            batch_images,
            session_encoder, session_decoder,
            save_path,
            max_instances, counted
        )
        ious.extend(batch_ious)
    
    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    print("\n=========== COCO Evaluation (Box Prompt) ===========")
    print(f"Instances Evaluated : {counted}")
    print(f"Mean IoU (mIoU)     : {miou:.4f}")
    print(f"Encoder Batch Size  : {encoder_batch_size}")
    print(f"Decoder Batch Size  : 1 (single instance)")
    print("====================================================\n")


def _process_batch(batch_images, session_encoder, session_decoder, save_path, 
                   max_instances, counted_start):
    """
    处理一个 batch 的推理和结果
    - Encoder: batch 推理 (多图一起编码)
    - Decoder: 单 instance 推理 (每个 instance 单独调用 infer)
    
    Returns:
        counted: 更新后的 instance 计数
        ious: 该 batch 的 IoU 列表
    """
    batch_size = len(batch_images)
    counted = counted_start
    ious = []
    
    # ========== 1. Encoder Batch Inference ==========
    input_images_batch = np.stack([img['image_preprocessed'] for img in batch_images], axis=0).astype(np.float32)
    high_res_feats_0, high_res_feats_1, image_embed = encoder_infer_batch(session_encoder, input_images_batch)
    
    # ========== 2. Decoder 逐个 Instance 推理 ==========
    for img_idx, img_data in enumerate(batch_images):
        img_masks = []
        
        for ann in img_data['anns']:
            if max_instances > 0 and counted >= max_instances:
                break
            
            box_xyxy = coco_bbox_to_xyxy(ann["bbox"])
            
            # 准备单个 instance 的 Decoder 输入 (batch=1)
            decoder_inputs = decoder_preprocessing(
                high_res_feats_0[img_idx].astype(np.float32),
                high_res_feats_1[img_idx].astype(np.float32),
                image_embed[img_idx].astype(np.float32),
                img_data['image_orig_hw'],
                box=box_xyxy
            )
            
            # 单个 instance 调用 infer
            masks, iou_predictions, low_res_masks = decoder_infer_single(session_decoder, decoder_inputs)
            # 插值到原始图像尺寸
            masks_resized = decoder_postprocessing(masks, img_data['image_orig_hw'][0], img_data['image_orig_hw'][1])
            masks_resized = masks_resized.squeeze(0).squeeze(0)
            pred2d = (masks_resized > 0).astype(np.uint8)
            img_masks.append(pred2d)
            # 计算 IoU
            gt_mask = rle_to_mask(ann["segmentation"], 
                                  img_data['image_orig_hw'][0], 
                                  img_data['image_orig_hw'][1])
            iou = compute_iou(pred2d, gt_mask)
            ious.append(iou)
            
            counted += 1
        
        # 保存可视化结果
        if save_path is not None and len(img_masks) > 0:
            save_mask_overlay(img_masks, img_data['image_orig'], 
                             save_path, img_data['img_info']["file_name"])
        
        if max_instances > 0 and counted >= max_instances:
            break
    
    return counted, ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, 
                        help='input path to coco dataset')
    parser.add_argument('--save_path', type=str, 
                        help='output path to image')
    parser.add_argument('--encoder_model_path', type=str, 
                        help='path to encoder model')
    parser.add_argument('--decoder_model_path', type=str, 
                        help='path to decoder model')
    parser.add_argument('--device-id', type=int, default=0,
                        help='NPU device id.')
    parser.add_argument('--bs', type=int, default=1, 
                        help='Batch size for encoder inference only (default: 1)')
    parser.add_argument('--max_instances', type=int, default=0, 
                        help='Maximum number of instances to evaluate (0 = all).')
    args = parser.parse_args()

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(os.path.realpath(args.save_path), mode=0o744)

    print(f"=== SAM2 COCO Evaluation ===")
    print(f"Encoder Model: {args.encoder_model_path}")
    print(f"Decoder Model: {args.decoder_model_path}")
    print(f"Encoder Batch Size: {args.bs}")
    print(f"Decoder Batch Size: 1 (single instance)")
    print(f"Device ID: {args.device_id}")
    print(f"Max Instances: {args.max_instances}")
    print("============================\n")

    session_encoder = InferSession(args.device_id, args.encoder_model_path)
    session_decoder = InferSession(args.device_id, args.decoder_model_path)

    evaluate_sam_on_coco(
        args.dataset_path,
        args.save_path,
        session_encoder,
        session_decoder,
        encoder_batch_size=args.bs,
        max_instances=args.max_instances
    )


if __name__ == "__main__":
    main()