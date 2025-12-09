# Copyright © 2021 - 2025. Huawei Technologies Co., Ltd. All Rights Reserved.
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

import numpy as np
import torch
from tqdm import tqdm

from concern.config import Configurable, Config

HIGHT_FIXED = 800


class Eval:
    def __init__(self, experiment, args, cmd):
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.data_loaders = experiment.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment.logger
        self.structure = experiment.structure

    @staticmethod
    def get_pred(filename):
        path_base = os.path.join(flags.bin_data_path, filename.split(".")[0])
        mask_pred = np.fromfile(path_base + "_" + '0' + ".bin", dtype="float32")
        width = mask_pred.shape[0] // HIGHT_FIXED
        mask_pred = np.reshape(mask_pred, [1, 1, HIGHT_FIXED, width])
        mask_pred = torch.from_numpy(mask_pred)
        return mask_pred

    def eval(self):
        for _, data_loader in self.data_loaders.items():
            raw_metrics = []
            for batch in tqdm(data_loader, total=len(data_loader)):
                pred = self.get_pred(batch['filename'][0])
                output = self.structure.representer.represent(batch, pred, is_output_polygon=True)  
                raw_metric = self.structure.measurer.validate_measure(
                    batch,
                    output, 
                    is_output_polygon=True,
                    box_thresh=self.args['box_thresh']
                    )
                raw_metrics.append(raw_metric)

            metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
            for key, metric in metrics.items():
                print('%s : %f (%d)' % (key, metric.avg, metric.count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--bin_data_path', default="./outputs/")
    parser.add_argument('--box_thresh', type=float, default=0.7,
                        help='The threshold to replace it in the representers')
    flags = parser.parse_args()

    global_args = parser.parse_args()
    global_args = vars(global_args)
    global_args = {k: v for k, v in global_args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(global_args['exp']))['Experiment']
    experiment_args.update(cmd=global_args)
    global_experiment = Configurable.construct_class_from_config(experiment_args)

    Eval(global_experiment, experiment_args, cmd=global_args).eval()