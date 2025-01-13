import os
import torch
import torch_npu
import torch.distributed as dist
from torch_npu._C._distributed_c10d import ProcessGroupHCCL


class ParallelManager:
    def __init__(self):
        local_rank = int(os.environ.get("LOCAL_RANK", "0")) 
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = local_rank
        self.world_size = world_size
        if self.world_size > 1:
            self.init_group()

    
    def init_group(self):
        device = torch.device(f"npu:{self.rank}")
        torch.npu.set_device(device)

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
    
