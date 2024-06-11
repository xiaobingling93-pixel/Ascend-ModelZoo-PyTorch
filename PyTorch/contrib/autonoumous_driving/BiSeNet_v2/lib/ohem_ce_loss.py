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

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


#  import ohem_cpp
#  class OhemCELoss(nn.Module):
#
#      def __init__(self, thresh, lb_ignore=255):
#          super(OhemCELoss, self).__init__()
#          self.score_thresh = thresh
#          self.lb_ignore = lb_ignore
#          self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='mean')
#
#      def forward(self, logits, labels):
#          n_min = labels[labels != self.lb_ignore].numel() // 16
#          labels = ohem_cpp.score_ohem_label(
#                  logits, labels, self.lb_ignore, self.score_thresh, n_min).detach()
#          loss = self.criteria(logits, labels)
#          return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh, lb_ignore=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.lb_ignore = lb_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        n_min = torch.ne(labels, self.lb_ignore).sum() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


if __name__ == '__main__':
    pass

