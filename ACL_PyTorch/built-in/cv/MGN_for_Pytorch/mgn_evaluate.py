# Copyright (c) 2025 Huawei Technologies Co., Ltd
# [Software Name] is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
import numpy as np
import torch
from scipy.spatial.distance import cdist

from MGN.opt import opt
from MGN.data import Data
from MGN.network import MGN
from MGN.utils.metrics import mean_ap, cmc, re_ranking

from om_executor import OMExcutor


def extract_feature_om(prediction_dir, prediction_flip_dir, output_idx='_0.txt'):

    def get_sorted_files(path, suffix):
        return sorted([
            fname for fname in os.listdir(path)
            if fname.endswith(suffix)
        ])

    # make the list of files first
    file_names = get_sorted_files(prediction_dir, output_idx)
    file_names_flip = get_sorted_files(prediction_flip_dir, output_idx)

    if len(file_names) != len(file_names_flip):
        raise ValueError("Mismatch in number of original and flipped feature files.")

    features = []
    for i, (fname, fname_flip) in enumerate(zip(file_names, file_names_flip)):
        f1 = torch.from_numpy(np.loadtxt(os.path.join(prediction_dir, fname), dtype=np.float32))
        f2 = torch.from_numpy(np.loadtxt(os.path.join(prediction_flip_dir, fname_flip), dtype=np.float32))

        ff = f1 + f2
        ff = ff.unsqueeze(0)
        ff = torch.nn.functional.normalize(ff, p=2, dim=1)
        features.append(ff)     # torch.cat会触发数据复制，效率较低，最后统一torch.cat一次，性能更好

        if i % 100 == 0:
            print(f"Extracted {i} features...")

    return torch.cat(features, dim=0)


class Evaluator(OMExcutor):
    def __init__(self, data):
        super().__init__(data)

    def evaluate_om(self):

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)
            return r, m_ap

        query_prediction_file_path = os.path.join(opt.result, "q_out")
        query_prediction_file_path_flip = os.path.join(opt.result, "q_filp")
        gallery_prediction_file_path = os.path.join(opt.result, "g_out")
        gallery_prediction_file_path_flip = os.path.join(opt.result, "g_filp")
        print('extract features, this may take a few minutes')
        qf = extract_feature_om(query_prediction_file_path, query_prediction_file_path_flip).numpy()
        gf = extract_feature_om(gallery_prediction_file_path, gallery_prediction_file_path_flip).numpy()

        # re rank
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        r, m_ap = rank(dist)
        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        # no re rank
        dist = cdist(qf, gf)
        r, m_ap = rank(dist)
        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

if __name__ == '__main__':
    data = Data()
    mgn_evaluator = Evaluator(data)
    print("start result evaluate")
    mgn_evaluator.evaluate_om()