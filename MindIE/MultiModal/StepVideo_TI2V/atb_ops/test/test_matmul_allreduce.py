import os
import time
from random import randint
import torch
import numpy as np
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from ops.matmul_allreduce import matmul_allreduce

WORLD_SIZE = 4


def test(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1111'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)

    input_tensor = torch.ones([10, 20], dtype=torch.bfloat16, device='npu')
    weight = torch.ones([20, 20], dtype=torch.bfloat16, device='npu')
    output_tensor = torch.empty_like(input_tensor, device='npu')
    output_tensor_check = torch.matmul(input_tensor, weight) * WORLD_SIZE

    matmul_allreduce(output_tensor, input_tensor, weight)

    if not torch.equal(output_tensor, output_tensor_check):
        print("Result check failed:")
        print(output_tensor)
    else:
        print(f"RANK{rank} Success!")


def _multiprocessing(world_size, func):
    ctx = mp.get_context('spawn')
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=func, args=(rank, world_size))
        p.start()

    for p in procs:
        p.join()

if __name__ == '__main__':
    _multiprocessing(WORLD_SIZE, test)