import math
import torch
import torch.distributed as dist

from typing import Any, Dict, List, Optional, Tuple, Union

from .parallel_mgr import ParallelManager

mgr = ParallelManager()


def get_world_size():
    return mgr.world_size


def get_rank():
    return mgr.rank


def get_dp_world_size():
    return mgr.dp_world_size


def get_dp_rank():
    return mgr.dp_rank


def get_sp_world_size():
    return mgr.sp_world_size


def get_sp_rank():
    return mgr.sp_rank


def get_sp_group():
    return mgr.sp_group


def get_dp_group():
    return mgr.dp_group


def all_gather(input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False, world_size=1, group=None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    if not (-input_.dim() <= dim < input_.dim()):
        raise ValueError(f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    input_size = list(input_.size())
    input_size[0] *= world_size
    output_tensor = torch.empty(
        input_size, dtype=input_.dtype, device=input_.device
    )
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=group
    )
    if dim != 0:
        input_size[0] //= world_size
        output_tensor = output_tensor.reshape([world_size, ] + input_size)
        output_tensor = output_tensor.movedim(0, dim)

    if separate_tensors:
        tensor_list = [
            output_tensor.view(-1)
            .narrow(0, input_.numel() * i, input_.numel())
            .view_as(input_)
            for i in range(world_size)
        ]
        return tensor_list
    else:
        input_size = list(input_.size())
        input_size[dim] = input_size[dim] * world_size
        # Reshape
        output_tensor = output_tensor.reshape(input_size)
        return output_tensor


def all_gather_variable_with_group(tensor, dim=0, world_size=1, group=None):
    """
    使用指定的 group 进行 all_gather 操作，支持第一维大小不同的张量。

    Args:
        tensor (torch.Tensor): 本地张量，第一维大小可能不同。
        dim (int): 拼接的维度, 默认是0。
        group (torch.distributed.ProcessGroup): 指定的进程组。

    Returns:
        torch.Tensor: 合并后的张量。
    """
    if world_size == 1:
        return tensor
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    # 获取当前张量的第一维大小
    local_size = torch.tensor([tensor.size(dim)], dtype=torch.long, device=tensor.device)
    
    # 收集所有进程的大小
    size_list = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    sizes = [int(size.item()) for size in size_list]
    
    # 找到最大大小
    max_size = max(sizes)
    
    # 如果当前张量小于最大大小，则填充
    if tensor.size(dim) < max_size:
        padding_size = list(tensor.size())
        padding_size[dim] = max_size - tensor.size(dim)
        padding = torch.zeros(*padding_size, dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat([tensor, padding], dim=dim)
    else:
        tensor_padded = tensor
    
    # 准备一个列表来存储所有填充后的张量
    tensor_list = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    
    # 执行 all_gather
    dist.all_gather(tensor_list, tensor_padded, group=group)
    
    # 去除填充并拼接
    tensors = []
    for i, t in enumerate(tensor_list):
        if sizes[i] > 0:
            tensors.append(t.narrow(dim, 0, sizes[i]))
    return torch.cat(tensors, dim=dim)


def split_tensor(input_tensor: torch.Tensor, dim: int, world_size: int, group: dist.ProcessGroup, scale=1):
    """
    将 input_tensor 沿指定维度 dim 切分为 group 中各个进程的部分。

    参数:
        input_tensor (torch.Tensor): 输入的张量。
        group (torch.distributed.ProcessGroup): 当前的通信组。
        dim (int): 切分的维度。

    返回:
        tuple:
            - torch.Tensor: 当前进程对应的切分后的张量。
            - int or None: 如果切分等长，返回切分后的长度；否则返回 None。
    """
    if world_size == 1:
        return input_tensor

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dim_size = input_tensor.size(dim)

    # 计算每个块的大小
    if dim_size / scale % world_size == 0:
        split_size = dim_size // world_size
    else:
        split_size = math.ceil(dim_size / world_size / 2) * 2

    chunks = torch.split(input_tensor, split_size, dim=dim)

    # 获取当前进程对应的块
    tensor_chunk = chunks[rank]

    return tensor_chunk
