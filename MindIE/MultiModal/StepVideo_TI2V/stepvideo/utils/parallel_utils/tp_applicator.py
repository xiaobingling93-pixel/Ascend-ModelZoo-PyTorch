import torch
import torch_npu
from xfuser.model_executor.models.customized.step_video_t2v.blocks import SelfAttention, CrossAttention, FeedForward
from xfuser.model_executor.models.customized.step_video_t2v.linear import ColumnParallelLinear, RowParallelLinear
from xfuser.core.distributed.parallel_state import get_tp_group
from stepvideo.text_encoder.stepllm import TransformerBlock
from stepvideo.parallel import enable_llm_tensor_model_parallel


class TensorParallelApplicator:
    def __init__(self, tp_size, tp_rank, device_map="cpu", tp_group=None):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group or get_tp_group()
        self.device_map = device_map

    def apply_to_model(self, model):
        self._apply_tp_to_attention(model)
        self._apply_tp_to_ffn(model)

    def apply_to_llm_model(self, model):
        self._replace_transformer_block(model)

    def _apply_tp_to_attention(self, module):
        for _, child in module.named_children():
            if child.__class__.__name__ == "SelfAttention":
                self._replace_self_attention(child)
            if child.__class__.__name__ == "CrossAttention":
                self._replace_cross_attention(child)
            else:
                self._apply_tp_to_attention(child)

    def _replace_transformer_block(self, module):
        for _, child in module.named_children():
            if isinstance(child, TransformerBlock):
                tp_size = self.tp_size
                tp_rank = self.tp_rank
                self._replace_llm_attention(child.attention, tp_rank, tp_size)

                self._replace_llm_ffn(child.feed_forward, tp_rank, tp_size)
            else:
                self._replace_transformer_block(child)

    def _replace_llm_attention(self, child, tp_rank, tp_size):
        orig_wqkv = child.wqkv
        orig_wo = child.wo

        column_out = orig_wqkv.out_features // tp_size
        row_in = orig_wo.in_features // tp_size

        child.wqkv = ColumnParallelLinear(
                orig_wqkv.in_features,
                column_out,
                bias=False,
                gather_output=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=self.tp_group
        ).to(orig_wqkv.weight.dtype).to(self.device_map)
        
        child.wo = RowParallelLinear(
                row_in,
                orig_wo.out_features,
                bias=False,
                input_is_parallel=True,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=self.tp_group
        ).to(orig_wo.weight.dtype).to(self.device_map)

        with torch.no_grad():
            kv_len = child.head_dim * 2 * child.n_groups
            wq_split, wkv_split = torch.split(
                orig_wqkv.weight, 
                (orig_wqkv.weight.shape[0] - kv_len,
                    kv_len                    
                ),
                dim=0,
            )
            wq_chunk = torch.chunk(wq_split, tp_size, dim=0)[tp_rank]
            wkv_chunk = torch.chunk(wkv_split, tp_size, dim=0)[tp_rank]
            wqkv_chunk = torch.cat([wq_chunk, wkv_chunk], dim=0)
            child.wqkv.weight.copy_(wqkv_chunk)
            wo_chunk = torch.chunk(orig_wo.weight, tp_size, dim=1)[tp_rank]
            child.wo.weight.copy_(wo_chunk)


    def _replace_llm_ffn(self, ff_layer, tp_rank, tp_size):
        orig_w1 = ff_layer.w1
        orig_w2 = ff_layer.w2

        ff_layer.w1 = ColumnParallelLinear(
                orig_w1.in_features,
                orig_w1.out_features // tp_size,
                bias=False,
                gather_output=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=self.tp_group
        ).to(orig_w1.weight.dtype).to(self.device_map)


        ff_layer.w2 = RowParallelLinear(
                orig_w2.in_features // tp_size,
                orig_w2.out_features,
                bias=False,
                input_is_parallel=True,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=self.tp_group
        ).to(orig_w2.weight.dtype).to(self.device_map)


        with torch.no_grad():
            w1_chunk = torch.chunk(orig_w1.weight, tp_size * 2, dim=0)[tp_rank::tp_size] # for swiglu
            w1_chunk = torch.cat(w1_chunk, dim=0)
            ff_layer.w1.weight.copy_(w1_chunk)

            w2_chunk = torch.chunk(orig_w2.weight, tp_size, dim=1)[tp_rank]
            ff_layer.w2.weight.copy_(w2_chunk)

    def _replace_self_attention(self, child):
        orig_wqkv = child.wqkv
        orig_wo = child.wo
        orig_dtype = orig_wqkv.weight.dtype
        column_out = orig_wqkv.out_features // self.tp_size
        row_in = orig_wo.in_features // self.tp_size

        child.wqkv = ColumnParallelLinear(
            orig_wqkv.in_features,
            column_out,
            bias=orig_wqkv.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.wo = RowParallelLinear(
            row_in,
            orig_wo.out_features,
            bias=orig_wo.bias is not None,
            input_is_parallel=True,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        self._split_self_weights(child, orig_wqkv, orig_wo)
        child.n_heads_per_tp = child.n_heads // self.tp_size

    def _split_self_weights(self, new_layer, orig_wqkv, orig_wo):
        wqkv_chunk = torch.chunk(orig_wqkv.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.wqkv.weight.data = wqkv_chunk.contiguous()

        wo_chunk = torch.chunk(orig_wo.weight.data, self.tp_size, dim=1)[self.tp_rank]
        new_layer.wo.weight.data = wo_chunk.contiguous()

        if orig_wqkv.bias is not None:
            bias_chunk = torch.chunk(orig_wqkv.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.wqkv.bias.data = bias_chunk.contiguous()
        if orig_wo.bias is not None:
            new_layer.wo.bias.data = orig_wo.bias.data.clone() / self.tp_size

    def _replace_cross_attention(self, child):
        orig_wq = child.wq
        orig_wkv = child.wkv
        orig_wo = child.wo
        orig_dtype = orig_wq.weight.dtype

        column_out_wq = orig_wq.out_features // self.tp_size
        column_out_wkv = orig_wkv.out_features // self.tp_size
        row_in_wo = orig_wo.in_features // self.tp_size

        child.wq = ColumnParallelLinear(
            orig_wq.in_features,
            column_out_wq,
            bias=orig_wq.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.wkv = ColumnParallelLinear(
            orig_wkv.in_features,
            column_out_wkv,
            bias=orig_wkv.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.wo = RowParallelLinear(
            row_in_wo,
            orig_wo.out_features,
            bias=orig_wo.bias is not None,
            input_is_parallel=True,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        self._split_cross_attention_weights(child, orig_wq, orig_wkv, orig_wo)
        child.n_heads_per_tp = child.n_heads // self.tp_size

    def _split_cross_attention_weights(self, new_layer, orig_wq, orig_wkv, orig_wo):
        wq_chunk = torch.chunk(orig_wq.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.wq.weight.data = wq_chunk.contiguous()
        if orig_wq.bias is not None:
            wq_bias_chunk = torch.chunk(orig_wq.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.wq.bias.data = wq_bias_chunk.contiguous()

        wkv_chunk = torch.chunk(orig_wkv.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.wkv.weight.data = wkv_chunk.contiguous()
        if orig_wkv.bias is not None:
            wkv_bias_chunk = torch.chunk(orig_wkv.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.wkv.bias.data = wkv_bias_chunk.contiguous()

        wo_chunk = torch.chunk(orig_wo.weight.data, self.tp_size, dim=1)[self.tp_rank]
        new_layer.wo.weight.data = wo_chunk.contiguous()
        if orig_wo.bias is not None:
            new_layer.wo.bias.data = orig_wo.bias.data.clone() / self.tp_size

    def _apply_tp_to_ffn(self, module):
        for _, child in module.named_children():
            if child.__class__.__name__ == "FeedForward":
                self._replace_ffn_layers(child)
            else:
                self._apply_tp_to_ffn(child)

    def _replace_ffn_layers(self, ff_layer):
        orig_gelu_linear = ff_layer.net[0].proj
        inner_dim_per_tp = orig_gelu_linear.out_features // self.tp_size
        orig_dtype = orig_gelu_linear.weight.dtype
        ff_layer.net[0].proj = ColumnParallelLinear(
            in_features=orig_gelu_linear.in_features,
            out_features=inner_dim_per_tp,
            bias=orig_gelu_linear.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        orig_output_linear = ff_layer.net[2]
        ff_layer.net[2] = RowParallelLinear(
            in_features=inner_dim_per_tp,
            out_features=orig_output_linear.out_features,
            bias=orig_output_linear.bias is not None,
            input_is_parallel=True,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        self._split_ffn_weights(ff_layer, orig_gelu_linear, orig_output_linear)

    def _split_ffn_weights(self, new_ffn, orig_first_linear, orig_second_linear):
        with torch.no_grad():
            first_weight_chunk = torch.chunk(orig_first_linear.weight.data, self.tp_size, dim=0)[self.tp_rank]
            new_ffn.net[0].proj.weight.data.copy_(first_weight_chunk.contiguous())

            if orig_first_linear.bias is not None:
                first_bias_chunk = torch.chunk(orig_first_linear.bias.data, self.tp_size, dim=0)[self.tp_rank]
                new_ffn.net[0].proj.bias.data.copy_(first_bias_chunk.contiguous())

            second_weight_chunk = torch.chunk(orig_second_linear.weight.data, self.tp_size, dim=1)[self.tp_rank]
            new_ffn.net[2].weight.data.copy_(second_weight_chunk.contiguous())

            if orig_second_linear.bias is not None:
                new_ffn.net[2].bias.data.copy_(orig_second_linear.bias.data.clone() / self.tp_size)
