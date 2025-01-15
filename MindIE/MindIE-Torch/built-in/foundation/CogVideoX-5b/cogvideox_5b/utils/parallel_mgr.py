import os
import torch
import torch_npu
import torch.distributed as dist
from torch_npu._C._distributed_c10d import ProcessGroupHCCL


def create_sp_group(world_size, rank):
    ranks = [i for i in range(world_size)]
    group1 = dist.new_group(ranks=ranks[:world_size // 2], backend='hccl')
    group2 = dist.new_group(ranks=ranks[world_size // 2:], backend='hccl')
    if rank < world_size // 2:
        subgroup = group1
    else:
        subgroup = group2
    print(f'rank: {rank}, ranks: {ranks}')
    return subgroup


def create_dp_group(world_size, rank):
    ranks = [i for i in range(world_size)]
    sub_ranks = [[i, j] for i, j in zip(ranks[:world_size // 2], ranks[world_size // 2:])]
    groups = [dist.new_group(ranks=sub_rank, backend='hccl') for sub_rank in sub_ranks]
    rank = rank if rank < world_size // 2 else rank - world_size // 2
    return groups[rank]


class ParallelManager:
    def __init__(self):
        local_rank = int(os.environ.get("LOCAL_RANK", "0")) 
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = local_rank
        self.world_size = world_size
        if self.world_size > 1:
            self.init_group()
        self.sp_group = None
        self.dp_group = None
        self.sp_rank = 0
        self.sp_world_size = 1
        self.dp_world_size = 1
        self.dp_rank = 0
        if self.world_size == 2:
            self.sp_rank = dist.get_rank(group=self.sp_group)
            self.sp_world_size = dist.get_world_size(group=self.sp_group)

        if self.world_size == 4 or self.world_size == 8:
            self.init_dp()
            self.dp_group = create_dp_group(self.world_size, self.rank)
            self.sp_group = create_sp_group(self.world_size, self.rank)
            self.sp_rank = dist.get_rank(group=self.sp_group)
            self.sp_world_size = dist.get_world_size(group=self.sp_group)

    def init_dp(self):
        self.dp_world_size = 2
        self.dp_rank = int(self.rank >= (self.world_size // self.dp_world_size))

    
    def init_group(self):
        device = torch.device(f"npu:{self.rank}")
        torch_npu.npu.set_device(device)

        backend = "hccl"
        options = ProcessGroupHCCL.Options()
        print("ProcessGroupHCCL has been Set")
        if not torch.distributed.is_initialized():
            # Call the init process.
            torch.distributed.init_process_group(
                backend=backend,
                world_size=self.world_size,
                rank=self.rank,
                pg_options=options,
            )
            print(f"rank {self.rank} init {torch.distributed.is_initialized()}, init_process_group has been activated")
        else:
            print("torch.distributed is already initialized.")
    
