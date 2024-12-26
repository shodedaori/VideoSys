import pytest

import torch

from videosys.models.modules.attentions import OpenSoraAttention, OpenSoraMultiHeadCrossAttention
from videosys.models.modules.cache_attn import OpenSoraAttention as CacheAttention
from videosys.models.modules.cache_attn import OpenSoraMultiHeadCrossAttention as CacheCrossAttention
from videosys.models.modules.cache_attn import QKVCache, PatchGather
from videosys.utils.test import empty_cache

from tests.fast_infer.utils import get_compute_device, tensor_check


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
def test_self_attn(device):
    dim = 128
    base_module = OpenSoraAttention(dim).to(device); base_module.eval()
    cache_module = CacheAttention(dim).to(device); cache_module.eval()
    
    for p_base, p_cache in zip(base_module.parameters(), cache_module.parameters()):
        p_cache.data.copy_(p_base.data)

    B, T, S, C = 2, 3, 5, dim
    x_input = torch.randn(B, T, S, C, device=device)
    x_base = x_input.view(B * T, S, C)
    x_cache = x_input.view(B, T * S, C)

    # test without cache
    y_output = base_module(x_base)
    y_base = y_output.view(B, T, S, C)

    y_output = cache_module(x_cache, T=T, S=S)
    y_cache = y_output.view(B, T, S, C)

    tensor_check(y_base, y_cache)

    # test cache without index
    cache = QKVCache(3, T, B, T * S, dim, dtype=x_input.dtype, device=device)
    y_output = cache_module(x_cache, T, S, cache=cache)
    y_cache = y_output.view(B, T, S, C)

    tensor_check(y_base, y_cache)

    # test cache with index
    index = torch.arange(T * S)
    index = index[torch.randperm(T * S)][:5].to(device)
    select_input = x_cache.index_select(1, index)
    y_cache = cache_module(select_input, T, S, cache=cache, token_index=index)

    y_pare = y_base.view(B, T * S, C)
    y_pare = y_pare.index_select(1, index)

    tensor_check(y_pare, y_cache)

    # test cache without index again
    y_output = cache_module(x_cache, T, S, cache=cache)
    y_cache = y_output.view(B, T, S, C)

    tensor_check(y_base, y_cache)


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
def test_cross_attn(device):
    dim = 128
    base_module = OpenSoraMultiHeadCrossAttention(dim, 8).to(device)
    cache_module = CacheCrossAttention(dim, 8).to(device)
    cache_module.load_state_dict(base_module.state_dict())

    B, T, S, C = 2, 3, 5, dim
    x_input = torch.randn(B, T * S, C, device=device)
    cond_input = torch.randn(B, 10, C, device=device)
    mask = [0, 0]


    # test without cache
    y_output = base_module(x_input, cond_input, mask=mask)
    y_base = y_output.view(B, T, S, C)

    y_output = cache_module(x_input, cond_input, T, S, mask=mask)
    y_cache = y_output.view(B, T, S, C)

    tensor_check(y_base, y_cache)

    # test cache without index
    cache = QKVCache(1, T, B, T * S, dim, dtype=x_input.dtype, device=device)
    y_output = cache_module(x_input, cond_input, T, S, mask=mask, cache=cache)
    y_cache = y_output.view(B, T, S, C)

    tensor_check(y_base, y_cache)

    # test cache with index
    index = torch.arange(T * S)
    index = index[torch.randperm(T * S)][:5].to(device)
    select_input = x_input.index_select(1, index)
    y_cache = cache_module(select_input, cond_input, T, S, mask=mask, cache=cache, token_index=index)
    
    y_pare = y_base.view(B, T * S, C)
    y_pare = y_pare.index_select(1, index)

    tensor_check(y_pare, y_cache)

    # test cache without index again
    y_output = cache_module(x_input, cond_input, T, S, mask=mask, cache=cache)
    y_cache = y_output.view(B, T, S, C)

    tensor_check(y_base, y_cache)


def test_patch_gather():
    x = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)  # (1, 1, 2, 2, 4)
    x = x.float()

    patch_gather = PatchGather(patch_size=(1, 2, 2))
    y = patch_gather(x)

    z = torch.tensor([[[14.],
         [22.],
         [46.],
         [54.]]])

    assert y.size() == (1, 4, 1)
    assert torch.equal(y, z)


if __name__ == "__main__":
    test_self_attn(torch.device("cpu"))
    test_cross_attn(torch.device('cuda'))
    test_patch_gather()
