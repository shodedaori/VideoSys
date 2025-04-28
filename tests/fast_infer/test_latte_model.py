import pytest

import torch
import torch.nn as nn
import torch.distributed as dist

from einops import rearrange, repeat
from videosys.models.transformers.latte_transformer_3d import BasicTransformerBlock, BasicTransformerBlock_
from videosys.models.modules.la_stu_attn import QKVCache
from videosys.models.modules.la_stu_block import BasicTransformerBlock as TestBlock
from videosys.models.modules.la_stu_block import BasicTransformerBlock_ as TestBlock_

from videosys.models.transformers.latte_transformer_3d import LatteT2V
from videosys.models.transformers.latte_stu_model import LatteT2V as TestModel

from videosys.utils.test import empty_cache
from videosys.core.parallel_mgr import ParallelManager

from tests.fast_infer.utils import tensor_check, init_env


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
@torch.no_grad()
def test_latte_block(device):
    base_block = BasicTransformerBlock(
        dim=1152,
        num_attention_heads=16,
        attention_head_dim=72,
        cross_attention_dim=1152,
        num_embeds_ada_norm=1000,
        attention_bias=True,
        norm_type='ada_norm_single',
        norm_eps=1e-6,
    ).to(device)

    test_block = TestBlock(
        dim=1152,
        num_attention_heads=16,
        attention_head_dim=72,
        cross_attention_dim=1152,
        num_embeds_ada_norm=1000,
        attention_bias=True,
        norm_type='ada_norm_single',
        norm_eps=1e-6,
    ).to(device)

    for bp, tp in zip(base_block.parameters(), test_block.parameters()):
        tp.data.copy_(bp.data)

    B, T, S, C = 2, 16, 1024, 1152
    hs = torch.randn(B, T, S, C, dtype=base_block.attn1.to_q.weight.dtype, device=device)
    base_hs = hs.view(B * T, S, C)
    test_hs = hs.view(B, T * S, C)

    encoder_hs = torch.randn(B * T, 6, C, dtype=hs.dtype, device=device)
    timestep = torch.randn(B, 6 * C, dtype=hs.dtype, device=device)
    timestep_spatial = repeat(timestep, "b d -> (b f) d", f=T).contiguous()
    stepindex = torch.tensor([2, 2], dtype=hs.dtype, device=device)

    # ========== Test without cache ==========

    base_out = base_block(
        hidden_states=base_hs,
        encoder_hidden_states=encoder_hs,
        timestep=timestep_spatial,
        org_timestep=stepindex
    )
    test_out = test_block(
        hidden_states=test_hs,
        encoder_hidden_states=encoder_hs,
        timestep=timestep,
        sub_batch_size=T,
        context_length=S
    )
    base_out = rearrange(base_out, "(b t) s c -> b (t s) c", b=B, t=T)

    tensor_check(base_out, test_out)

    # ========== Test with cache and None index ==========

    att1_cache = QKVCache(3, B, T, S, C, dtype=hs.dtype, device=device)
    att2_cache = QKVCache(1, B, T, S, C, dtype=hs.dtype, device=device)
    qkv_cache = (att1_cache, att2_cache)
    token_index = None

    test_out = test_block(
        hidden_states=test_hs,
        encoder_hidden_states=encoder_hs,
        timestep=timestep,
        attn_cache=qkv_cache,
        token_index=token_index,
        sub_batch_size=T,
        context_length=S
    )

    tensor_check(base_out, test_out)

    # ========== Test with cache and index ==========
    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    index_out = test_block(
        hidden_states=index_hs,
        encoder_hidden_states=encoder_hs,
        timestep=timestep,
        attn_cache=qkv_cache,
        token_index=token_index,
        sub_batch_size=T,
        context_length=S
    )
    index_base = base_out.index_select(1, token_index)

    tensor_check(index_base, index_out)

    # ========== Test with cache and fused projections ==========
    test_block.attn1.fuse_projections()

    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    index_out = test_block(
        hidden_states=index_hs,
        encoder_hidden_states=encoder_hs,
        timestep=timestep,
        attn_cache=qkv_cache,
        token_index=token_index,
        sub_batch_size=T,
        context_length=S
    )
    index_base = base_out.index_select(1, token_index)

    tensor_check(index_base, index_out)


    torch.cuda.synchronize(device)
    print("Spatial block test passed!")


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
@torch.no_grad()
def test_latte_block_temproal(device):
    base_block = BasicTransformerBlock_(
        dim=1152,
        num_attention_heads=16,
        attention_head_dim=72,
        num_embeds_ada_norm=1000,
        attention_bias=True,
        norm_type='ada_norm_single',
        norm_eps=1e-6,
    ).to(device)

    parallel_manager = ParallelManager(1, 1, 1)
    base_block.parallel_manager = parallel_manager

    test_block = TestBlock_(
        dim=1152,
        num_attention_heads=16,
        attention_head_dim=72,
        num_embeds_ada_norm=1000,
        attention_bias=True,
        norm_type='ada_norm_single',
        norm_eps=1e-6,
    ).to(device)

    for bp, tp in zip(base_block.parameters(), test_block.parameters()):
        tp.data.copy_(bp.data)

    B, T, S, C = 2, 16, 1024, 1152
    hs = torch.randn(B, S, T, C, dtype=base_block.attn1.to_q.weight.dtype, device=device)
    base_hs = hs.view(B * S, T, C)
    test_hs = hs.view(B, S * T, C)

    timestep = torch.randn(B, 6 * C, dtype=hs.dtype, device=device)
    timestep_temporal = repeat(timestep, "b d -> (b f) d", f=S).contiguous()
    stepindex = torch.tensor([2, 2], dtype=hs.dtype, device=device)

    # ========== Test without cache ==========

    base_out = base_block(
        hidden_states=base_hs,
        encoder_hidden_states=None,
        timestep=timestep_temporal,
        org_timestep=stepindex
    )

    test_out = test_block(
        hidden_states=test_hs,
        encoder_hidden_states=None,
        timestep=timestep,
        sub_batch_size=S,
        context_length=T
    )
    base_out = rearrange(base_out, "(b s) t c -> b (s t) c", b=B, s=S)

    tensor_check(base_out, test_out)

    # ========== Test with cache and None index ==========

    att1_cache = QKVCache(3, B, S, T, C, dtype=hs.dtype, device=device)
    qkv_cache = att1_cache
    token_index = None

    test_out = test_block(
        hidden_states=test_hs,
        encoder_hidden_states=None,
        timestep=timestep,
        attn_cache=qkv_cache,
        token_index=token_index,
        sub_batch_size=S,
        context_length=T
    )

    tensor_check(base_out, test_out)

    # ========== Test with cache and index ==========
    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    test_out = test_block(
        hidden_states=index_hs ,
        encoder_hidden_states=None,
        timestep=timestep,
        attn_cache=qkv_cache,
        token_index=token_index,
        sub_batch_size=S,
        context_length=T
    )
    index_base = base_out.index_select(1, token_index)

    tensor_check(index_base, test_out)

    # ========== Test with cache and fused projections ==========
    test_block.attn1.fuse_projections()

    n_index = int(0.5 * T * S)
    token_index = torch.randperm(T * S)[:n_index].to(device)
    index_hs = test_hs.index_select(1, token_index)
    
    test_out = test_block(
        hidden_states=index_hs,
        encoder_hidden_states=None,
        timestep=timestep,
        attn_cache=qkv_cache,
        token_index=token_index,
        sub_batch_size=S,
        context_length=T
    )
    index_base = base_out.index_select(1, token_index)

    tensor_check(index_base, test_out)


    torch.cuda.synchronize(device)
    print("Temporal block test passed!")


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@empty_cache
@torch.no_grad()
def test_latte_model(device):
    depth = 4
    init_kwargs = {
        "num_attention_heads": 16,
        "attention_head_dim": 72,
        "in_channels": 4,
        "out_channels": 8,
        "num_layers": depth,
        "cross_attention_dim": 1152,
        "attention_bias": True,
        "sample_size": 64,
        "patch_size": 2,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000,
        "use_linear_projection": False,
        "only_cross_attention": False,
        "double_self_attention": False,
        "upcast_attention": False,
        "norm_type": "ada_norm_single",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-6,
        "attention_type": "default",
        "caption_channels": 4096,
        "video_length": 16
    }

    base_model = LatteT2V(**init_kwargs).to(device).eval()
    base_model.enable_parallel(1, 1, 0)
    test_model = TestModel(**init_kwargs).to(device).eval()
    test_model.enable_parallel(1, 1, 0)

    for bp, tp in zip(base_model.parameters(), test_model.parameters()):
        tp.data.copy_(bp.data)

    B, T, C, H, W = 2, 16, 4, 64, 64
    S, D = 1024, 1152
    hs = torch.randn(B, C, T, H, W, dtype=base_model.pos_embed.proj.weight.dtype, device=device)
    encoder_hs = torch.randn(B, 6, 4096, dtype=hs.dtype, device=device)
    timestep = torch.tensor([2, 2], dtype=hs.dtype, device=device)

    test_model.set_2d_temp_embed(hs)

    # ========== Test without cache ==========

    base_out = base_model(
        hidden_states=hs,
        timestep=timestep,
        encoder_hidden_states=encoder_hs,
        return_dict=False,
    )[0]

    test_out = test_model(
        hidden_states=hs,
        timestep=timestep,
        encoder_hidden_states=encoder_hs,
        return_dict=False,
    )[0]

    tensor_check(base_out, test_out)

    # ========== Test with cache and None index ==========
    cache_list = []
    for i in range(depth):
        att1_cache = QKVCache(3, B, T, S, D, dtype=hs.dtype, device=device)
        att2_cache = QKVCache(1, B, T, S, D, dtype=hs.dtype, device=device)
        att3_cache = QKVCache(3, B, S, T, D, dtype=hs.dtype, device=device)
        cache_list.append((att3_cache, (att1_cache, att2_cache)))
    test_model.set_token_level_cache(hs)

    token_index = (None, None)

    test_out = test_model(
        hidden_states=hs,
        timestep=timestep,
        encoder_hidden_states=encoder_hs,
        cache_list=cache_list,
        token_index=token_index,
        return_dict=False,
    )[0]

    tensor_check(base_out, test_out)

    # ========== Test with cache and index ==========
    n_index = int(0.5 * T * S)
    spatial_index = torch.randperm(T * S)[:n_index].to(device)
    pos_a = spatial_index // S
    pos_b = spatial_index % S
    temporal_index = pos_b * T + pos_a
    token_index = (temporal_index, spatial_index)
    
    test_out = test_model(
        hidden_states=hs,
        timestep=timestep,
        encoder_hidden_states=encoder_hs,
        cache_list=cache_list,
        token_index=token_index,
        return_dict=False,
    )[0]

    tensor_check(base_out, test_out)

    # ========== Test with cache and fused projections ==========
    for i in range(depth):
        test_model.transformer_blocks[i].attn1.fuse_projections()
        test_model.temporal_transformer_blocks[i].attn1.fuse_projections()

    n_index = int(0.5 * T * S)
    spatial_index = torch.randperm(T * S)[:n_index].to(device)
    pos_a = spatial_index // S
    pos_b = spatial_index % S
    temporal_index = pos_b * T + pos_a
    token_index = (temporal_index, spatial_index)
    
    test_out = test_model(
        hidden_states=hs,
        timestep=timestep,
        encoder_hidden_states=encoder_hs,
        cache_list=cache_list,
        token_index=token_index,
        return_dict=False,
    )[0]

    tensor_check(base_out, test_out)


    torch.cuda.synchronize(device)
    print("Latte model test passed!")



if __name__ == "__main__":
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False

    init_env()
    
    # test_latte_block(torch.device('cuda'))
    # test_latte_block_temproal(torch.device('cuda'))
    test_latte_model(torch.device('cuda'))
