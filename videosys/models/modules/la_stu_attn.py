from typing import Optional

from einops import rearrange
import inspect
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


class QKVCache:
    def __init__(self, length, batch_size, sub_batch_size, context_length, hidden_dim, dtype, device):
        self.length = length
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        self.sequence_length = sub_batch_size * context_length

        self.cache = torch.empty(length, batch_size, self.sequence_length, hidden_dim, dtype=dtype, device=device)

    @property
    def shape(self):
        # The shape is supposed to be (BS, SUB, CONTEXT, HIDDEN)
        return (self.batch_size, self.sub_batch_size, self.context_length, self.hidden_dim)

    @property
    def packed(self):
        ret = [self.cache[i].view(self.shape) for i in range(self.length)]
        return ret

    def update(self, values, token_index=None):
        # value: (B, N, C)
        if token_index is None:
            for i in range(self.length):
                self.cache[i].copy_(values[i])
        else:
            for i in range(self.length):
                self.cache[i].index_copy_(1, token_index, values[i])

        return self.packed


class LatteAttnProcessor(AttnProcessor2_0):

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kvcache: Optional[QKVCache] = None,
        token_index: Optional[torch.Tensor] = None,
        sub_batch_size: Optional[int] = None,
        context_length: Optional[int] = None,
        *args, **kwargs
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            raise ValueError("Diffusers AttentionProcessor2_0 does not support args or kwargs")

        residual = hidden_states
        input_ndim = hidden_states.ndim
        assert input_ndim == 3

        batch_size = hidden_states.size(0)

        assert attention_mask is None
        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     # scaled_dot_product_attention expects attention_mask shape to be
        #     # (batch, heads, source_length, target_length)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        assert attn.group_norm is None
        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        self_attn_flag = encoder_hidden_states is None
        if attn.fused_projections:
            B, N, C = hidden_states.shape
            qkv = attn.to_qkv(hidden_states)
            query, key, value = qkv.view(B, N, 3, C).permute(2, 0, 1, 3).unbind(0)
        else:
            query = attn.to_q(hidden_states)

            if self_attn_flag:
                encoder_hidden_states = hidden_states
            else:
                assert attn.norm_cross is None

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if self_attn_flag:  # self-attention
            if kvcache is not None:  # update cache
                query, key, value = kvcache.update([query, key, value], token_index=token_index)  # [B, SUB, CONTEXT, HIDDEN]
            else:
                query = query.view(batch_size, sub_batch_size, context_length, -1)
                key = key.view(batch_size, sub_batch_size, context_length, -1)
                value = value.view(batch_size, sub_batch_size, context_length, -1)

            query = query.view(batch_size * sub_batch_size, context_length, -1)  # [B * SUB, CONTEXT, HIDDEN]
            key = key.view(batch_size * sub_batch_size, context_length, -1)  # [B * SUB, CONTEXT, HIDDEN]
            value = value.view(batch_size * sub_batch_size, context_length, -1)  # [B * SUB, CONTEXT, HIDDEN]

        else: # cross-attention
            if kvcache is not None:
                query = kvcache.update([query], token_index=token_index)[0]  # [B, SUB, CONTEXT, HIDDEN]
            else:
                query = query.view(batch_size, sub_batch_size, context_length, -1)

            query = query.view(batch_size * sub_batch_size, context_length, -1)  # [B * SUB, CONTEXT, HIDDEN]

        parallel_size = batch_size * sub_batch_size
        attn_heads = attn.heads
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_heads

        query = query.view(parallel_size, -1, attn_heads, head_dim).transpose(1, 2)  # [B * SUB, HEADS, CONTEXT, HIDDEN]
        key = key.view(parallel_size, -1, attn_heads, head_dim).transpose(1, 2)  # [B * SUB, HEADS, CONTEXT, HIDDEN]
        value = value.view(parallel_size, -1, attn_heads, head_dim).transpose(1, 2)  # [B * SUB, HEADS, CONTEXT, HIDDEN]

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )  # [B * SUB, HEADS, CONTEXT, HIDDEN]

        hidden_states = rearrange(hidden_states, "(b s) heads c h -> b (s c) (heads h)", b=batch_size, s=sub_batch_size)
        if token_index is not None:
            hidden_states = hidden_states.index_select(1, token_index)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class STUAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
