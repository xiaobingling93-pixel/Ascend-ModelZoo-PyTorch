__all__ = ["matmul_allreduce"]

import torch
import torch_npu
from ..op_builder.matmul_allreduce_builder import MatmulAllreduceOpBuilder

matmul_allreduce_builder = MatmulAllreduceOpBuilder()


def get_ranks_worldsize_info():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return rank, world_size


def matmul_allreduce(output, input, weight):
    matmul_allreduce_op = matmul_allreduce_builder.load()
    rank, ranksize = get_ranks_worldsize_info()
    matmul_allreduce_op.matmul_allreduce(input, weight, output, rank, ranksize)