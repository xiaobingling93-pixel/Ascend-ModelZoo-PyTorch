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
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.runner.checkpoint import find_latest_checkpoint

@RUNNERS.register_module()
class FCNRunner(Runner):

    def load_or_resume(self) -> None:
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if self._resume and self._load_from is None:
            # auto resume from the latest checkpoint
            resume_from = find_latest_checkpoint(self.work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            # resume from the specified checkpoint
            resume_from = self._load_from

        map_location = 'default'
        if 'LOCAL_RANK' in os.environ:
            map_location = 'npu:'+str(os.environ['LOCAL_RANK'])

        if resume_from is not None:
            self.resume(resume_from, map_location=map_location)
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from, map_location=map_location)
            self._has_loaded = True
