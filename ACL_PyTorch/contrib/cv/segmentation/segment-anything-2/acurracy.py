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
import torch
import torch_npu
import numpy as np
import cv2
from sklearn.metrics import jaccard_score
from scipy.ndimage import binary_erosion, binary_dilation
from tqdm import tqdm
from sam2.build_sam import build_sam2_camera_predictor

# 设置设备
torch.autocast(device_type="npu", dtype=torch.float16)
device = torch.device("npu")
torch_npu.npu.set_compile_mode(jit_compile=False)


def compute_iou(gt_mask, pred_mask):
    """计算交并比 (IoU)"""
    gt = gt_mask.flatten()
    pred = pred_mask.flatten()
    return jaccard_score(gt, pred)


def compute_boundary_f(gt_mask, pred_mask, tolerance=2):
    """计算边界 F 分数"""
    gt_boundary = get_boundary(gt_mask)
    pred_boundary = get_boundary(pred_mask)

    gt_dil = binary_dilation(gt_boundary, iterations=tolerance)
    pred_dil = binary_dilation(pred_boundary, iterations=tolerance)

    gt_match = gt_boundary & pred_dil
    pred_match = pred_boundary & gt_dil

    precision = pred_match.sum() / max(pred_boundary.sum(), 1)
    recall = gt_match.sum() / max(gt_boundary.sum(), 1)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_boundary(mask):
    """获取掩码边界"""
    eroded = binary_erosion(mask)
    boundary = mask - eroded
    return boundary.astype(bool)


def get_bbox_from_mask(mask):
    """从掩码中提取边界框"""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return np.array([[0, 0], [1, 1]], dtype=np.float32)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return np.array([[x_min, y_min], [x_max, y_max]], dtype=np.float32)


def evaluate_video(video_name, predictor, data_root):
    """评估视频序列"""
    print(f"评估视频: {video_name}")
    
    img_dir = os.path.join(data_root, "JPEGImages/480p", video_name)
    mask_dir = os.path.join(data_root, "Annotations_unsupervised/480p", video_name)

    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

    print(f"图像帧数: {len(img_paths)}, 掩码帧数: {len(mask_paths)}")
    
    j_scores = []
    f_scores = []

    for idx, (img_path, mask_path) in enumerate(tqdm(zip(img_paths, mask_paths), total=len(img_paths), desc="处理中", leave=False)):
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        if idx == 0:
            predictor.load_first_frame(frame)
            bbox = get_bbox_from_mask(gt_mask)
            predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox)
        else:
            _, mask_logits = predictor.track(frame)
            pred_mask = (mask_logits[0] > 0).squeeze().cpu().numpy().astype(np.uint8)

            j = compute_iou(gt_mask, pred_mask)
            f = compute_boundary_f(gt_mask, pred_mask)

            j_scores.append(j)
            f_scores.append(f)

    mean_j = np.mean(j_scores)
    mean_f = np.mean(f_scores)
    mean_jf = (mean_j + mean_f) / 2

    print("\n评估结果:")
    print(f"平均 J: {mean_j:.4f}")
    print(f"平均 F: {mean_f:.4f}")
    print(f"J&F: {mean_jf:.4f}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SAM2 视频分割评估')
    
    parser.add_argument('--data_path', type=str, default='../DAVIS',
                       help='数据集路径')
    parser.add_argument('--vdo_name', type=str, default='bear',
                       help='视频名称')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam2.1_hiera_small.pt',
                       help='模型检查点')
    parser.add_argument('--model_config', type=str, default='configs/sam2.1/sam2.1_hiera_s.yaml',
                       help='模型配置')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print(f"数据路径: {args.data_path}")
    print(f"视频名称: {args.vdo_name}")
    
    # 初始化预测器
    predictor = build_sam2_camera_predictor(args.model_config, args.checkpoint, device=device)
    
    # 评估视频
    evaluate_video(
        video_name=args.vdo_name,
        predictor=predictor,
        data_root=args.data_path
    )

if __name__ == "__main__":
    main()