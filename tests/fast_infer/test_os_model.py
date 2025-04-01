import pytest

import torch
import torch.nn as nn
import torch.distributed as dist

from videosys.core.parallel_mgr import ParallelManager, initialize
from videosys.models.transformers.open_sora_transformer_3d import STDiT3Config, STDiT3  
from videosys.models.transformers.os_tau_model import STDiT3C, OpenSoraSI
from videosys.models.modules.os_tau_attn import QKVCache
from videosys.utils.test import empty_cache

from tests.fast_infer.utils import tensor_check, init_env


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


class TestModel(nn.Module):
    def __init__(self, block_type, hidden_size, num_heads, depth):
        super().__init__()
        blocks = [block_type(hidden_size, num_heads, temporal=bool(i % 2)) for i in range(depth)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, y, t, mask, x_mask, t0, T, S):
        for block in self.blocks:
            x = block(x, y, t, mask, x_mask, t0, T=T, S=S)
        return x
    
    def new_forward(self, x, y, t, mask, T, S, cache=None, index=(None, None)):
        for k, block in enumerate(self.blocks):
            if cache:
                this_cache = cache[k]
            else:
                this_cache = (None, None)
            x = block(x, y, t, 
                    mask=mask, 
                    T=T, 
                    S=S, 
                    cache=this_cache,
                    index=index)
        return x


@torch.no_grad()
def test_block(device):
    from videosys.models.transformers.open_sora_transformer_3d import STDiT3Block
    from videosys.models.transformers.os_tau_model import STDiT3BlockC
    depth = 4

    base_module = TestModel(STDiT3Block, 128, 8, depth).to(device); base_module.eval()
    cache_module = TestModel(STDiT3BlockC, 128, 8, depth).to(device); cache_module.eval()
    cache_module.load_state_dict(base_module.state_dict())

    mgr = ParallelManager(1, 1, 1)
    for block in base_module.blocks:
        block.parallel_manager = mgr

    B, T, S, C = 2, 7, 3 * 5, 128
    L = 24
    mask = [12, 12]
    x = torch.randn(B, T * S, C, device=device)
    y = torch.randn(1, L, C, device=device)
    t = torch.randn(B, 6 * C, device=device)
    t0 = torch.randn(B, 6 * C, device=device)
    x_mask = torch.ones(B, T, device=device).bool()

    cache_module.blocks[-1].init_cache(B, T, S)

    z_base = base_module(x, y, t, mask, x_mask, t0, T, S)
    z_cache = cache_module.new_forward(x, y, t, mask, T, S)

    tensor_check(z_base, z_cache)
    ############################################################

    cache_list = []
    dtype = x.dtype
    for i in range(depth):
        if i % 2 == 0:
            self_attn_cache = QKVCache(3, T, B, T * S, C, dtype=dtype, device=device)
        else:
            self_attn_cache = QKVCache(3, S, B, T * S, C, dtype=dtype, device=device)
        cross_attn_cache = QKVCache(1, T, B, T * S, C, dtype=dtype, device=device)
        cache_list.append((self_attn_cache, cross_attn_cache))
    
    z_cache = cache_module.new_forward(x, y, t, mask, T, S, cache=cache_list)
    tensor_check(z_base, z_cache)
    ############################################################

    index = torch.arange(T * S)
    spatial_index = index[torch.randperm(T * S)][:50].to(device)
    pos_b = spatial_index % S
    pos_a = spatial_index // S
    temporal_index = pos_b * T + pos_a

    select_input = x.index_select(1, spatial_index)
    index = (temporal_index, spatial_index)
    z_cache = cache_module.new_forward(select_input, y, t, mask, T, S, cache=cache_list, index=index)

    tensor_check(z_base, z_cache)
    ############################################################

    z_cache = cache_module.new_forward(x, y, t, mask, T, S, cache=cache_list)
    tensor_check(z_base, z_cache)


@torch.no_grad()
def test_dit(device):
    config = STDiT3Config(
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        hidden_size=128,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        caption_channels=512,
        model_max_length=100,
    )

    base_model = STDiT3(config).to(device); base_model.eval()
    cache_model = STDiT3C(config).to(device); cache_model.eval()
    cache_model.load_state_dict(base_model.state_dict())

    base_model.enable_parallel(1, 1, 1)

    C, T, H, W = 4, 7, 17, 23
    L = 12
    height = torch.tensor([480.], device=device)
    width = torch.tensor([640.], device=device)
    fps = torch.tensor([24.], device=device)

    x_o = torch.randn(1, C, T, H, W, device=device)
    x = torch.cat([x_o, x_o], dim=0)
    t = torch.tensor([131.], device=device)
    y = torch.randn(1, 100, 512, device=device)
    mask = torch.zeros(1, 100, device=device).int()
    mask[0, :L] = 1
    x_mask = torch.ones(2, T, device=device).bool()

    z_base = base_model(x, t, None, y, mask, x_mask, fps=fps, height=height, width=width)
    z_cache = cache_model(x, t, y, mask, fps=fps, height=height, width=width)
    tensor_check(z_base, z_cache)
    ############################################################

    infer = OpenSoraSI(cache_model)
    infer.init_generate_cache(x_o)

    z_cache = infer(x, t, y, mask, fps=fps, height=height, width=width)
    tensor_check(z_base, z_cache)
    ############################################################

    index = infer.select_index(x_o, 0.5)
    z_cache = infer(x, t, y, mask, fps=fps, height=height, width=width, index=index)
    tensor_check(z_base, z_cache)
    ############################################################

    z_cache = cache_model(x, t, y, mask, fps=fps, height=height, width=width)
    tensor_check(z_base, z_cache)
    ############################################################
 

if __name__ == "__main__":
    
    # test_dit()
    # test_base(1)
    init_env()
    test_block(torch.device('cuda'))
    test_dit(torch.device('cuda'))
    dist.destroy_process_group()
