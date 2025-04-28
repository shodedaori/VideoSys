import pytest

import torch

from einops import rearrange
from diffusers.models.attention_processor import Attention
from videosys.models.modules.la_stu_attn import QKVCache, LatteAttnProcessor
from videosys.utils.test import empty_cache

from tests.fast_infer.utils import get_compute_device, tensor_check


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
def test_self_attn(device):
    base_attn = Attention(
        query_dim=1152,
        heads=16,
        dim_head=72,
        bias=True,
        upcast_attention=False,
        out_bias=True
    ).to(device)

    test_attn = Attention(
        query_dim=1152,
        heads=16,
        dim_head=72,
        bias=True,
        upcast_attention=False,
        out_bias=True,
        processor=LatteAttnProcessor()
    ).to(device)

    for bp, tp in zip(base_attn.parameters(), test_attn.parameters()):
        tp.data.copy_(bp.data)

    B, T, S, C = 2, 16, 1024, 1152
    hs = torch.randn(B, T, S, C, dtype=base_attn.to_q.weight.dtype, device=device)
    base_hs = hs.view(B * T, S, C)
    test_hs = hs.view(B, T * S, C)

    # ========== Test without cache ==========

    base_out = base_attn(base_hs, encoder_hidden_states=None, attention_mask=None)
    test_out = test_attn(test_hs, encoder_hidden_states=None, attention_mask=None, sub_batch_size=T, context_length=S)
    base_out = rearrange(base_out, "(b t) s c -> b (t s) c", b=B, t=T)

    tensor_check(base_out, test_out)

    # ========== Test with cache and None index ==========

    qkv_cache = QKVCache(3, B, T, S, C, dtype=hs.dtype, device=device)
    token_index = None

    test_out = test_attn(test_hs, encoder_hidden_states=None, attention_mask=None, kvcache=qkv_cache, token_index=token_index, sub_batch_size=T, context_length=S)

    tensor_check(base_out, test_out)

    # ========== Test with cache and index ==========
    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    index_out = test_attn(index_hs, encoder_hidden_states=None, attention_mask=None, kvcache=qkv_cache, token_index=token_index, sub_batch_size=T, context_length=S)
    index_base = base_out.index_select(1, token_index)

    tensor_check(index_base, index_out)

    # ========== Test with cache and fused projections ==========
    test_attn.fuse_projections()

    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    index_out = test_attn(index_hs, encoder_hidden_states=None, attention_mask=None, kvcache=qkv_cache, token_index=token_index, sub_batch_size=T, context_length=S)
    index_base = base_out.index_select(1, token_index)
    
    tensor_check(index_base, index_out)

    torch.cuda.synchronize(device)
    print("Self-attention test passed!")


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
def test_cross_attn(device):
    base_attn = Attention(
        query_dim=1152,
        heads=16,
        dim_head=72,
        bias=True,
        upcast_attention=False,
        out_bias=True
    ).to(device)

    test_attn = Attention(
        query_dim=1152,
        heads=16,
        dim_head=72,
        bias=True,
        upcast_attention=False,
        out_bias=True,
        processor=LatteAttnProcessor()
    ).to(device)

    B, T, S, C = 2, 16, 1024, 1552
    hs = torch.randn(B, T, S, C, dtype=base_attn.to_q.proj.weight.dtype, device=device)
    encoder_hs = torch.randn(B * T, 6, C, dtype=hs.dtype, device=device)
    base_hs = hs.view(B * T, S, C)
    test_hs = hs.view(B, T * S, C)

    # ========== Test without cache ==========

    base_out = base_attn(base_hs, encoder_hidden_states=encoder_hs, attention_mask=None)
    test_out = test_attn(test_hs, encoder_hidden_states=encoder_hs, attention_mask=None, sub_batch_size=T, context_length=S)
    base_out = rearrange(base_out, "(b t) s c -> b (t s) c", b=B, t=T)

    tensor_check(base_out, test_out)

    # ========== Test with cache and None index ==========

    qkv_cache = QKVCache(1, B, T, S, C, dtype=hs.dtype, device=device)
    token_index = None

    test_out = test_attn(test_hs, encoder_hidden_states=encoder_hs, attention_mask=None, kvcache=qkv_cache, token_index=token_index, sub_batch_size=T, context_length=S)

    tensor_check(base_out, test_out)

    # ========== Test with cache and index ==========
    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    index_out = test_attn(index_hs, encoder_hidden_states=encoder_hs, attention_mask=None, kvcache=qkv_cache, token_index=token_index, sub_batch_size=T, context_length=S)
    index_base = base_out.index_select(1, token_index)

    tensor_check(index_base, index_out)

    # ========== Test with cache and fused projections ==========
    test_attn.fuse_projections()

    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    index_out = test_attn(index_hs, encoder_hidden_states=encoder_hs, attention_mask=None, kvcache=qkv_cache, token_index=token_index, sub_batch_size=T, context_length=S)
    index_base = base_out.index_select(1, token_index)
    
    tensor_check(index_base, index_out)

    torch.cuda.synchronize(device)
    print("Cross-attention test passed!")


if __name__ == "__main__":
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
    
    # test_self_attn(torch.device("cuda"))
    test_cross_attn(torch.device("cuda"))
