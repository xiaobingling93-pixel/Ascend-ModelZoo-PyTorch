import torch.distributed as dist

_GLOBAL_PARALLEL_GROUPS = dict()

_SEQUENCE_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None

def set_data_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("data", None)


def set_sequence_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


def get_sequence_parallel_group_for_send_recv_overlap(check_initialized=True):
    """Get the sequence parallel group for send-recv overlap the caller rank belongs to."""
    if check_initialized:
        assert (
                _SEQUENCE_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP is not None
        ), 'context parallel group for send-recv overlap is not initialized'
    return _SEQUENCE_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP


def initialize_sequence_parallel_group_for_send_recv_overlap(
        use_cp_send_recv_overlap,
        sequence_parallel_size
):
    if use_cp_send_recv_overlap != True:
        return

    rank = dist.get_rank()
    world_size: int = dist.get_world_size()
    data_parallel_size: int = world_size // sequence_parallel_size
    global _SEQUENCE_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP

    for i in range(data_parallel_size):
        start_rank = i * sequence_parallel_size
        end_rank = (i + 1) * sequence_parallel_size
        ranks = range(start_rank, end_rank)
        group_send_recv_overlap = dist.new_group(ranks)

        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = group_send_recv_overlap
