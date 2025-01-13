import torch

from typing import Any, Dict, List, Optional, Tuple, Union

from .parallel_mgr import ParallelManager

mgr = ParallelManager()


def get_world_size():
    return mgr.world_size


def get_rank():
    return mgr.rank


def all_gather(input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    world_size = get_world_size()
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
        output_tensor, input_
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
