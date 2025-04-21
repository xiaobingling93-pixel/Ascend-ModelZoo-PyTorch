import os
import torch
import torch_npu
from einops import rearrange

from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from yunchang.ring.utils import RingComm, update_out_and_lse
from yunchang.ring.ring_flash_attn import RingFlashAttnFunc

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None
    from yunchang.kernels.attention import pytorch_attn_forward
import mindiesd
from mindiesd.layers.flash_attn.attention_forward import attention_forward


def xdit_ring_flash_attn_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        attn_layer=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="none",
):
    is_joint = False
    if (joint_tensor_key is not None and
            joint_tensor_value is not None):
        supported_joint_strategy = ["front", "rear"]
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
            )
        else:
            is_joint = True
    elif (joint_tensor_key is None and
          joint_tensor_value is None):
        pass
    else:
        raise ValueError(
            f"joint_tensor_key and joint_tensor_value should be None or not None simultaneously."
        )

    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    algo = int(os.getenv('ALGO', 0))

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        key, value = k, v
        if not causal or step <= comm.rank:
            if flash_attn is None:
                if algo == 0:
                    block_out = attention_forward(
                        q,
                        key,
                        value,
                        opt_mode="manual",
                        op_type="fused_attn_score",
                        layout="BNSD"
                    )
                elif algo == 1:
                    block_out = attention_forward(
                        q,
                        key,
                        value,
                        opt_mode="manual",
                        op_type="ascend_laser_attention",
                        layout="BNSD"
                    )

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    return block_out


class xFuserRingFlashAttnFunc(RingFlashAttnFunc):
    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_softmax,
            group,
            attn_layer,
            joint_tensor_key,
            joint_tensor_value,
            joint_strategy,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        # out, softmax_lse = xdit_ring_flash_attn_forward(
        out = xdit_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        return out


def xdit_ring_flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        group=None,
        attn_layer=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="none",
):
    return xFuserRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )
