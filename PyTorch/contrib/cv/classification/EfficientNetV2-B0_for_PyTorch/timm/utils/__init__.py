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

from .agc import adaptive_clip_grad
from .checkpoint_saver import CheckpointSaver
from .clip_grad import dispatch_clip_grad
from .cuda import ApexScaler, NativeScaler
from .decay_batch import decay_batch_step, check_batch_size_retry
from .distributed import distribute_bn, reduce_tensor, init_distributed_device,\
    world_info_from_env, is_distributed_env, is_primary
from .jit import set_jit_legacy, set_jit_fuser
from .log import setup_default_logging, FormatterNoInfo
from .metrics import AverageMeter, accuracy
from .misc import natural_key, add_bool_arg, ParseKwargs
from .model import unwrap_model, get_state_dict, freeze, unfreeze, reparameterize_model
from .model_ema import ModelEma, ModelEmaV2, ModelEmaV3, ModelEmaV2Npu
from .random import random_seed
from .summary import update_summary, get_outdir
