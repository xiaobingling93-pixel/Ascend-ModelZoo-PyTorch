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
import psutil
from loguru import logger


def _get_pcie_info(devices, keyword="PCIeBusInfo"):
    device_pcie_tbl = dict()
    for device in devices:
        pcie_info = os.popen(f"npu-smi info -t board -i {device}").read().strip().split("\n")
        for _ in pcie_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                device_pcie_tbl[device] = line[len(keyword) + 1:]
                break
    return device_pcie_tbl


def _get_numa_info(pcie_tbl, keyword="NUMAnode"):
    device_numa_tbl = dict()
    numa_devices_tbl = dict()
    for device, pcie_no in pcie_tbl.items():
        numa_info = os.popen(f"lspci -s {pcie_no} -vvv").read().strip().split("\n")
        for _ in numa_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1:])
                device_numa_tbl[device] = numa_id
                devices = numa_devices_tbl.get(numa_id, None)
                if devices is None:
                    numa_devices_tbl[numa_id] = list()
                numa_devices_tbl[numa_id].append(device)
                break
    return device_numa_tbl, numa_devices_tbl


def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    cpu_info = os.popen(f"lscpu").read().strip().split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")
            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise Exception("lscpu command output error, please check !")
                ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]
            numa_id = int(split_info[0].replace(keyword1, '').replace(keyword2, ''))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl


def bind_cpus(ratio=0.5):
    visible_devices = "0,1,2,3,4,5,6,7"
    if visible_devices is None:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        devices = [_ for _ in range(world_size)]
    else:
        devices = list(map(int, visible_devices.split(',')))
    device_pcie_tbl = _get_pcie_info(devices)
    device_numa_tbl, numa_devices_tbl = _get_numa_info(device_pcie_tbl)
    cpu_idx_tbl = _get_cpu_info(list(numa_devices_tbl.keys()))
    rank_id = int(os.environ["RANK"])
    cur_device = devices[rank_id]
    numa_id = device_numa_tbl[cur_device]
    shard_devices = numa_devices_tbl[numa_id]
    shard_devices.sort()
    all_cpus = cpu_idx_tbl[numa_id]
    logger.info(
        f"rank_id: {rank_id}, device_id: {cur_device}, numa_id: {numa_id}, shard_devices: {shard_devices}, cpus: {all_cpus}")
    cpu_nums = len(all_cpus)
    CPU_BINDING_NUM = os.environ.get("CPU_BINDING_NUM", None)
    if CPU_BINDING_NUM is None:
        cpu_num_per_device = int(cpu_nums * ratio // len(shard_devices))
    else:
        cpu_num_per_device = int(CPU_BINDING_NUM)
        if len(shard_devices) * cpu_num_per_device > cpu_nums:
            raise Exception(
                f"Cpu num in numa {numa_id} to assign {cpu_num_per_device} for every device is not enough, "
                f"please decrease the value of CPU_BINDING_NUM!")
    idx = shard_devices.index(cur_device)
    binding_cpus = [all_cpus[_] for _ in range(idx * cpu_num_per_device, (idx + 1) * cpu_num_per_device)]
    p = psutil.Process()
    p.cpu_affinity(binding_cpus)
    new_affinity = p.cpu_affinity()
    logger.info(f"process {p.pid}, new_affinity is {new_affinity}, cpu count {cpu_num_per_device}")
