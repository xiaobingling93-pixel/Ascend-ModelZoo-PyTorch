import torch.distributed as dist
import xfuser
import torch

_LLM_TP_ENABLE = False


def initialize_parall_group(ring_degree, ulysses_degree, tensor_parallel_degree, llm_tensor_parallel_degree=None):
    dist.init_process_group("hccl")
    xfuser.core.distributed.init_distributed_environment(
        rank=dist.get_rank(), 
        world_size=dist.get_world_size(),
        backend="hccl"
    )
    
    xfuser.core.distributed.initialize_model_parallel(
        sequence_parallel_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
        tensor_parallel_degree=tensor_parallel_degree,
    )
    torch.npu.set_device(dist.get_rank())


def enable_llm_tensor_model_parallel():
    global _LLM_TP_ENABLE
    _LLM_TP_ENABLE = True


def get_llm_tensor_model_parallel_world_size():
    return dist.get_world_size() if _LLM_TP_ENABLE else 1
    

def get_llm_tensor_model_parallel_rank():
    return dist.get_rank() if _LLM_TP_ENABLE else 0


def get_llm_tensor_model_parallel_group():
    return dist.group.WORLD if _LLM_TP_ENABLE else None


def get_parallel_group():
    return xfuser.core.distributed.get_world_group()


def get_sequence_parallel_world_size():
    return xfuser.core.distributed.parallel_state.get_sequence_parallel_world_size()


def get_sequence_parallel_rank():
    return xfuser.core.distributed.parallel_state.get_sequence_parallel_rank()


def get_sp_group():
    return xfuser.core.distributed.parallel_state.get_sp_group()


def parallel_forward(fn_):
    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        if kwargs['parallel']:            
            hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            kwargs['attn_mask'] = torch.chunk(kwargs['attn_mask'], get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        output = fn_(_, hidden_states, *args, **kwargs)
        
        if kwargs['parallel']:
            output = get_sp_group().all_gather(output.contiguous(), dim=-2)
        
        return output
     
    return wrapTheFunction
