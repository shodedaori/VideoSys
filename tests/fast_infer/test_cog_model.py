import pytest

import torch
import torch.nn as nn
import torch.distributed as dist

from videosys.core.parallel_mgr import ParallelManager, initialize
from videosys.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel
from videosys.models.transformers.cog_tau_model import KVCache, CogTauBlock, CogTauTransformer3DModel, CogVideoSTU
from videosys.utils.test import empty_cache

from tests.fast_infer.utils import tensor_check, init_env


class FakeManager:
    def __init__(self):
        self.sp_size = 1


class TestModel(nn.Module):
    def __init__(self, block_type, dim, num_attention_heads, attention_head_dim, time_embed_dim, depth):
        super().__init__()
        blocks = [block_type(dim, num_attention_heads, attention_head_dim, time_embed_dim) for i in range(depth)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, y, t):
        for block in self.blocks:
            x, y = block(x, y, t)
        return x, y
    
    def new_forward(self, x, y, t, cache=None, index=None):
        for k, block in enumerate(self.blocks):
            if cache:
                this_cache = cache[k]
            else:
                this_cache = None
            x, y = block(x, y, t, kvcache=this_cache, token_index=index)
        return x, y


@torch.no_grad()
def test_block(device):
    depth = 1

    base_module = TestModel(CogVideoXBlock, 128, 8, 32, 64, depth).to(device); base_module.eval()
    cache_module = TestModel(CogTauBlock, 128, 8, 32, 64, depth).to(device); cache_module.eval()
    cache_module.load_state_dict(base_module.state_dict())

    mgr = ParallelManager(1, 1, 1)
    for block in base_module.blocks:
        block.parallel_manager = mgr
        block.attn1.parallel_manager = mgr

    B, L, T, C = 2, 3, 5, 128
    x = torch.randn(B, T, C, device=device)
    y = torch.randn(B, L, C, device=device)
    t = torch.randn(B, 64, device=device)

    cache_module.blocks[-1].set_input_cache(L, T, C)

    z_base = base_module(x, y, t)
    z_cache = cache_module.new_forward(x, y, t)

    tensor_check(z_base[0], z_cache[0])
    tensor_check(z_base[1], z_cache[1])
    ############################################################

    cache_list = []
    dtype = x.dtype
    for i in range(depth):
        cache = KVCache(2, L, T, 8 * 32, dtype=dtype, device=device)
        cache_list.append(cache)

    z_cache = cache_module.new_forward(x, y, t, cache=cache_list)
    
    tensor_check(z_base[0], z_cache[0])
    tensor_check(z_base[1], z_cache[1])
    ############################################################

    index = torch.arange(T)
    index = index[torch.randperm(T)][:2].to(device)
    sx = x.index_select(1, index)
    z_cache = cache_module.new_forward(sx, y, t, cache=cache_list, index=index)

    tensor_check(z_base[0], z_cache[0])
    tensor_check(z_base[1], z_cache[1])
    ############################################################

    z_cache = cache_module.new_forward(x, y, t, cache=cache_list)
    tensor_check(z_base[0], z_cache[0])
    tensor_check(z_base[1], z_cache[1])

    torch.cuda.synchronize()
    print("Block test passed")
 

@torch.no_grad()
def test_model(device):
    config = {
        "num_attention_heads": 16,
        "attention_head_dim": 64,
        "time_embed_dim": 512,
        "text_embed_dim": 4096,
        "num_layers": 4
    }
    dim = config["num_attention_heads"] * config["attention_head_dim"]
    width = 90
    height = 60
    frames = 13

    base_model = CogVideoXTransformer3DModel(**config).to(device); base_model.eval()
    cache_model = CogTauTransformer3DModel(**config).to(device); cache_model.eval()
    cache_model.load_state_dict(base_model.state_dict())

    base_model.enable_parallel(1, 1, 1)

    L, T, H, W, C = 226, width * height * frames // 4, height, width, 16
    x_o = torch.randn(1, frames, C, H, W, device=device)
    x = torch.cat([x_o, x_o], dim=0)
    y = torch.randn(2, L, config["text_embed_dim"], device=device)
    t = torch.tensor([131., 131.], device=device)

    z_base = base_model(x, y, t)[0]
    z_cache = cache_model(x, y, t)[0]

    tensor_check(z_base, z_cache)
    ############################################################
    infer = CogVideoSTU(cache_model)
    infer.init_generate_cache(x_o, L)

    z_cache = infer(x, y, t, use_cache=True)[0]
    tensor_check(z_base, z_cache)
    ############################################################
    index = torch.arange(T)
    Tp = int(0.5 * T)
    index = index[torch.randperm(T)][:Tp].to(device)
    z_cache = infer(x, y, t, use_cache=True, token_index=index)[0]
    tensor_check(z_base, z_cache)
    ############################################################

    z_cache = infer(x, y, t, use_cache=True)[0]
    tensor_check(z_base, z_cache)
    ############################################################


if __name__ == "__main__":
    
    # test_dit()
    # test_base(1)
    init_env()
    # test_block(torch.device('cuda'))
    test_model(torch.device('cuda'))
    dist.destroy_process_group()
