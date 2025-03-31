import pytest

import torch
import time
from tqdm import tqdm

from diffusers.models.attention import Attention
from videosys.models.transformers.cogvideox_transformer_3d import CogVideoXAttnProcessor2_0
from videosys.models.transformers.cog_tau_model import KVCache, CogTauAttnProcessor2_0
from videosys.utils.test import empty_cache

from tests.fast_infer.utils import get_compute_device, tensor_check

class FakeManager:
    def __init__(self):
        self.sp_size = 1


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
def test_self_attn(device):
    dim = 512
    n_heads = 8
    dim_heads = dim // n_heads    

    base_attn = Attention(
        query_dim=dim,
        dim_head=dim_heads,
        heads=n_heads,
        qk_norm="layer_norm",
        eps=1e-6,
        bias=False,
        out_bias=True,
        processor=CogVideoXAttnProcessor2_0(),
    ).to(device)
    base_attn.eval()
    base_attn.parallel_manager = FakeManager()

    test_attn = Attention(
        query_dim=dim,
        dim_head=dim_heads,
        heads=n_heads,
        qk_norm="layer_norm",
        eps=1e-6,
        bias=False,
        out_bias=True,
        processor=CogTauAttnProcessor2_0(),
    ).to(device)
    test_attn.eval()

    
    for p_base, p_cache in zip(base_attn.parameters(), test_attn.parameters()):
        p_cache.data.copy_(p_base.data)

    B, L, T, C = 2, 2, 8, dim
    x_input = torch.randn(B, T, C, device=device)
    y_input = torch.randn(B, L, C, device=device)
    kvcache = KVCache(2, L, T, dim, dtype=x_input.dtype, device=device)

    # test without cache
    base_output = base_attn(hidden_states=x_input, encoder_hidden_states=y_input)
    test_output = test_attn(hidden_states=x_input, encoder_hidden_states=y_input)

    tensor_check(base_output[0], test_output[0])
    tensor_check(base_output[1], test_output[1])

    # test cache without index
    test_output = test_attn(hidden_states=x_input, encoder_hidden_states=y_input, kvcache=kvcache)

    tensor_check(base_output[0], test_output[0])
    tensor_check(base_output[1], test_output[1])

    # test cache with index
    index = torch.arange(T)
    index = index[torch.randperm(T)][:4].to(device)
    sx_input = x_input.index_select(1, index)
    test_output = test_attn(hidden_states=sx_input, encoder_hidden_states=y_input, kvcache=kvcache, token_index=index)

    tensor_check(base_output[0].index_select(1, index), test_output[0])
    tensor_check(base_output[1], test_output[1])


    # test cache without index again
    test_output = test_attn(hidden_states=x_input, encoder_hidden_states=y_input, kvcache=kvcache)

    tensor_check(base_output[0], test_output[0])
    tensor_check(base_output[1], test_output[1])


    # test cache with index and rotary embedding
    sin = torch.randn(T, dim_heads, device=device, dtype=torch.float)
    cos = torch.randn(T, dim_heads, device=device, dtype=torch.float)
    rotary_emb = (sin, cos)

    base_output = base_attn(hidden_states=x_input, encoder_hidden_states=y_input, image_rotary_emb=rotary_emb)
    test_output = test_attn(hidden_states=sx_input, encoder_hidden_states=y_input, image_rotary_emb=rotary_emb, kvcache=kvcache, token_index=index)

    tensor_check(base_output[0].index_select(1, index), test_output[0])
    tensor_check(base_output[1], test_output[1])

    torch.cuda.synchronize()
    print("Test passed")


def profile_attn(device):
    dim = 4096
    n_heads = 64
    dim_heads = dim // n_heads    

    base_attn = Attention(
        query_dim=dim,
        dim_head=dim_heads,
        heads=n_heads,
        qk_norm="layer_norm",
        eps=1e-6,
        bias=False,
        out_bias=True,
        processor=CogVideoXAttnProcessor2_0(),
    ).to(device)
    base_attn.eval()
    base_attn.parallel_manager = FakeManager()

    test_attn = Attention(
        query_dim=dim,
        dim_head=dim_heads,
        heads=n_heads,
        qk_norm="layer_norm",
        eps=1e-6,
        bias=False,
        out_bias=True,
        processor=CogTauAttnProcessor2_0(),
    ).to(device)
    test_attn.eval()

    for p_base, p_cache in zip(base_attn.parameters(), test_attn.parameters()):
        p_cache.data.copy_(p_base.data)

    B, L, T, C = 1, 226, 1550, dim
    x_input = torch.randn(B, T, C, device=device)
    y_input = torch.randn(B, L, C, device=device)

    test_time = 10
    warm_time = 3

    for _ in range(warm_time):
        base_output = base_attn(hidden_states=x_input, encoder_hidden_states=y_input)
        s = base_output[0].sum()

    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(test_time)):
        base_output = base_attn(hidden_states=x_input, encoder_hidden_states=y_input)
        s = base_output[0].sum()
    torch.cuda.synchronize()
    # print average time in ms
    print(f"Base time: {(time.time() - start) / test_time * 1000:.2f} ms")

    kvcache = KVCache(2, L, T, dim, dtype=x_input.dtype, device=device)
    for _ in range(warm_time):
        test_output = test_attn(hidden_states=x_input, encoder_hidden_states=y_input, kvcache=kvcache)
        s = test_output[0].sum()

    index = torch.arange(T)
    part_T = int(T * 0.5)
    index = index[torch.randperm(T)][:part_T].to(device)
    sx_input = x_input.index_select(1, index)

    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(test_time)):
        test_output = test_attn(hidden_states=sx_input, encoder_hidden_states=y_input, kvcache=kvcache, token_index=index)
        s = test_output[0].sum()
    torch.cuda.synchronize()
    # print average time in ms
    print(f"Test time: {(time.time() - start) / test_time * 1000:.2f} ms")


if __name__ == "__main__":
    test_self_attn(torch.device("cuda"))
    # profile_attn(torch.device("cuda"))
