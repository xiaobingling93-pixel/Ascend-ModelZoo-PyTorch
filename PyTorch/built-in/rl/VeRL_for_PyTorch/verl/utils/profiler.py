# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

# Copied from https://gitee.com/ascend/MindSpeed-RL/blob/master/mindspeed_rl/utils/utils.py
import os
from functools import wraps

import torch
import torch_npu


def get_grpo_profiler(profiler_config, role: str = None):
    if not profiler_config or not profiler_config.profile:
        return None

    profiler_this_rank = False
    if profiler_config.profile_ranks == "all":
        profiler_this_rank = True
    else:
        try:
            ranks = list(profiler_config.profile_ranks)
        except (TypeError, AttributeError):
            ranks = [0]
        if (torch.distributed.get_rank() in ranks):
            profiler_this_rank = True
    if not profiler_this_rank:
        return None

    if profiler_config.profile_level == 'level_none':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level_none
    elif profiler_config.profile_level == 'level0':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level0
    elif profiler_config.profile_level == 'level1':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level1
    elif profiler_config.profile_level == 'level2':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        raise ValueError(f"profiler_level only supports level0,"
                         f" 1, 2, and level_none, but gets {profiler_config.profile_level}")

    if profiler_config.profile_export_type == 'text':
        profile_export_type = torch_npu.profiler.ExportType.Text
    elif profiler_config.profile_export_type == 'db':
        profile_export_type = torch_npu.profiler.ExportType.Db
    else:
        raise ValueError(f"profile_export_type only supports text or db,"
                         f"but gets {profiler_config.export_type}")

    base_path = profiler_config.profile_save_path
    if role:
        profile_save_path = os.path.join(base_path, role)
    else:
        profile_save_path = base_path

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=profiler_level,
        export_type=profile_export_type,
        data_simplification=True,
        msprof_tx=profiler_config.mstx
    )
    if profiler_config.stage == "all":
        skip_first = profiler_config.profile_step_start
        active = profiler_config.profile_step_end - profiler_config.profile_step_start
    else:
        skip_first = 0
        active = 1

    activites = []
    if profiler_config.profile_with_npu:
        activites.append(torch_npu.profiler.ProfilerActivity.NPU)
    if profiler_config.profile_with_cpu:
        activites.append(torch_npu.profiler.ProfilerActivity.CPU)

    prof = torch_npu.profiler.profile(
        with_modules=profiler_config.profile_with_module,
        record_shapes=profiler_config.profile_record_shapes,
        profile_memory=profiler_config.profile_with_memory,
        activities=activites,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path,
                                                                    analyse_flag=profiler_config.profile_analysis),
        experimental_config=experimental_config)

    return prof


def mstx_timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        range_id = torch_npu.npu.mstx.range_start(func.__qualname__)
        result = func(*args, **kwargs)
        torch_npu.npu.mstx.range_end(range_id)
        return result
    return wrapper


def profiler_start(profiler_config, role="profiler_data", profiler_iteration=None):
    if not profiler_config:
        return None
    if profiler_iteration is not None and (
            profiler_iteration < profiler_config.profile_step_start or
            profiler_iteration >= profiler_config.profile_step_end):
        return None
    if profiler_config.stage == "all" and role != profiler_config.role:
        return None
    if profiler_config.stage != "all" and role != profiler_config.stage:
        return None
    profiler = get_grpo_profiler(profiler_config, role)
    if not profiler:
        return None
    profiler.start()
    return profiler


def profiler_step(profiler):
    if profiler:
        profiler.step()
