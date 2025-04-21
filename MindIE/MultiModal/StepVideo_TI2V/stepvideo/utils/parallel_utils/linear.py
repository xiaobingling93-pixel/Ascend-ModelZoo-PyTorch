import torch
import torch.nn as nn
import torch_npu
from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_tensor_model_parallel_world_size
)
from xfuser.core.distributed.group_coordinator import GroupCoordinator


class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gather_output=True, tp_size=None, tp_rank=None, tp_group=None):
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank or get_tensor_model_parallel_rank()
        self.tp_group = tp_group or get_tp_group()

        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        x = super().forward(x)
        return x


class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True, tp_size=None, tp_rank=None, tp_group=None, matmul_allreduce_type='atb'):
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank or get_tensor_model_parallel_rank()
        self.tp_group = tp_group or get_tp_group()
        self.input_is_parallel = input_is_parallel

        if matmul_allreduce_type == 'atb':
            try:
                from atb_ops.ops.matmul_allreduce import matmul_allreduce
                self.matmul_allreduce = matmul_allreduce
                self.matmul_allreduce_type = "atb"
            except Exception:
                self.matmul_allreduce = None
                self.matmul_allreduce_type = "torch"
        else:
            self.matmul_allreduce_type = matmul_allreduce_type

        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        if not self.input_is_parallel:
            x = torch.chunk(x, self.tp_size, dim=-1)[self.tp_rank]
        
        # x, b,s,h1; w h1, h2
        if self.matmul_allreduce_type == "atb":
            if x.dim() == 2:
                output = torch.empty((x.shape[0], self.weight.shape[0]), dtype=x.dtype, device=x.device)
            elif x.dim() == 3:
                b, s, hx = x.size()
                output = torch.empty((b, s, self.weight.shape[0]), dtype=x.dtype, device=x.device)    
            self.matmul_allreduce(output, x, self.weight)
        elif self.matmul_allreduce_type == "torch_npu":
            if isinstance(self.tp_group, GroupCoordinator):
                tp_pg = self.tp_group.device_group
            else:
                tp_pg = self.tp_group
            hcom = tp_pg._get_backend(torch.device('npu')).get_hccl_comm_name
            output = torch_npu.npu_mm_all_reduce_base(x, self.weight, hcom)
        else:
            x = super().forward(x)
            # 执行All-Reduce聚合
            if isinstance(self.tp_group, GroupCoordinator):
                output = self.tp_group.all_reduce(x)
            else:
                torch.distributed.all_reduce(x, group=self.tp_group)
                output = x

        return output
