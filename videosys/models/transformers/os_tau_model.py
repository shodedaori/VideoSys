# Adapted from OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


from collections.abc import Iterable
import numpy as np
import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from transformers import PretrainedConfig, PreTrainedModel

from videosys.core.comm import all_to_all_with_pad, gather_sequence, get_pad, set_pad, split_sequence
from videosys.core.parallel_mgr import ParallelManager
from videosys.models.modules.activations import approx_gelu
# from videosys.models.modules.attentions import OpenSoraAttention, OpenSoraMultiHeadCrossAttention
from videosys.models.modules.embeddings import (
    OpenSoraCaptionEmbedder,
    OpenSoraPatchEmbed3D,
    OpenSoraPositionEmbedding2D,
    SizeEmbedder,
    TimestepEmbedder,
)
from videosys.utils.utils import batch_func
from videosys.models.modules.os_tau_attn import QKVCache, OpenSoraAttention, OpenSoraMultiHeadCrossAttention
from videosys.models.modules.rotary import RotaryEmbedding
from videosys.models.transformers.utils import PatchGather, ShorttermWindow
from videosys.models.transformers.stu_model import STUBase


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)


class STDiT3BlockC(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        block_idx=None,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = OpenSoraAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )
        self.cross_attn = OpenSoraMultiHeadCrossAttention(hidden_size, num_heads, enable_flash_attn=enable_flash_attn)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # parallel
        self.parallel_manager: ParallelManager = None
        self.block_idx = block_idx
        self.input_cache = None


    def init_cache(self, B, T, S):
        dtype = self.mlp.fc1.weight.dtype
        device = self.mlp.fc1.weight.device
        self.input_cache = QKVCache(
            1, T, B, T * S, self.hidden_size, 
            dtype=dtype, device=device)

    def update_cache(self, x, token_index=None):
        y = self.input_cache.update([x], token_index)[0]
        y = y.view(self.input_cache.bs, -1, self.hidden_size)
        return y

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
        cache=(None, None),  # the cache used in self-attention and cross-attention
        index=(None, None),  # the index used in temporal attention and spatial attention
    ):
        assert x_mask is None, "The mask should be None for now"
        # if this block is the last block, fill the input
        self_attn_cache, cross_attn_cache = cache
        temporal_index, spatial_index = index

        if self.input_cache and self_attn_cache is not None:
            x = self.update_cache(x, spatial_index)
            self_attn_cache = cross_attn_cache = None
            temporal_index = spatial_index = None
        
        B, N, C = x.shape
        input_partial_flag = (N < T * S)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)
        
        # attention
        if self.temporal:
            # if self.parallel_manager.sp_size > 1:
            #     x_m, S, T = self.dynamic_switch(x_m, S, T, to_spatial_shard=True)
            
            if not input_partial_flag:
                x_m = rearrange(x_m, "b (t s) d -> b (s t) d", t=T, s=S)
            x_m = self.attn(x_m, S, T, cache=self_attn_cache, token_index=temporal_index)
            if not input_partial_flag:
                x_m = rearrange(x_m, "b (s t) d -> b (t s) d", t=T, s=S)
            
            # if self.parallel_manager.sp_size > 1:
            #     x_m, S, T = self.dynamic_switch(x_m, S, T, to_spatial_shard=False)
        else:
            # x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(x_m, T, S, cache=self_attn_cache, token_index=spatial_index)
            # x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        x_cross = self.cross_attn(x, y, T, S, mask, cache=cross_attn_cache, token_index=spatial_index)
        x = x + x_cross

        # print("x shape", x.shape, x[0][104][0:5])

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        return x

    def dynamic_switch(self, x, s, t, to_spatial_shard: bool):
        raise NotImplementedError("dynamic_switch is not implemented in STDiT3Block right now")
        # if to_spatial_shard:
        #     scatter_dim, gather_dim = 2, 1
        #     scatter_pad = get_pad("spatial")
        #     gather_pad = get_pad("temporal")
        # else:
        #     scatter_dim, gather_dim = 1, 2
        #     scatter_pad = get_pad("temporal")
        #     gather_pad = get_pad("spatial")
        # x = rearrange(x, "b (t s) d -> b t s d", t=t, s=s)
        # x = all_to_all_with_pad(
        #     x,
        #     self.parallel_manager.sp_group,
        #     scatter_dim=scatter_dim,
        #     gather_dim=gather_dim,
        #     scatter_pad=scatter_pad,
        #     gather_pad=gather_pad,
        # )
        # new_s, new_t = x.shape[2], x.shape[1]
        # x = rearrange(x, "b t s d -> b (t s) d")
        # return x, new_s, new_t


class STDiT3Config(PretrainedConfig):
    model_type = "STDiT3"

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=False,
        only_train_temporal=False,
        freeze_y_embedder=False,
        skip_y_embedder=False,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.only_train_temporal = only_train_temporal
        self.freeze_y_embedder = freeze_y_embedder
        self.skip_y_embedder = skip_y_embedder
        super().__init__(**kwargs)


class STDiT3C(PreTrainedModel):
    config_class = STDiT3Config

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = OpenSoraPositionEmbedding2D(config.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self.x_embedder = OpenSoraPatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        self.y_embedder = OpenSoraCaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3BlockC(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3BlockC(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                    block_idx=i,
                )
                for i in range(config.depth)
            ]
        )
        self.temporal_blocks[-1].attn.partial_output_flag = False
        # final layer
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()
        if config.only_train_temporal:
            for param in self.parameters():
                param.requires_grad = False
            for block in self.temporal_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        if config.freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False

        # parallel
        self.parallel_manager: ParallelManager = None

        # cache
        self.input_cache = None

    def enable_parallel(self, dp_size, sp_size, enable_cp):
        # update cfg parallel
        if enable_cp and sp_size % 2 == 0:
            sp_size = sp_size // 2
            cp_size = 2
        else:
            cp_size = 1

        self.parallel_manager = ParallelManager(dp_size, cp_size, sp_size)

        for name, module in self.named_modules():
            if "spatial_blocks" in name or "temporal_blocks" in name:
                if hasattr(module, "parallel_manager"):
                    module.parallel_manager = self.parallel_manager

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def init_cache(self, B, T, S):
        dtype = self.x_embedder.proj.weight.dtype
        device = self.x_embedder.proj.weight.device
        cache_hs = np.prod(self.patch_size) * self.out_channels
        self.input_cache = QKVCache(1, T, B, T * S, cache_hs, dtype=dtype, device=device)

    def forward(
        self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, 
        cache=None, index=(None, None), *args, **kwargs
    ):
        assert x_mask is None, "The mask should be None for now"
        # === Split batch ===
        # if self.parallel_manager.cp_size > 1:
        #     x, timestep, y, x_mask, mask = batch_func(
        #         partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0),
        #         x,
        #         timestep,
        #         y,
        #         x_mask,
        #         mask,
        #     )

        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # shard over the sequence dim if sp is enabled
        # if self.parallel_manager.sp_size > 1:
        #     set_pad("temporal", T, self.parallel_manager.sp_group)
        #     set_pad("spatial", S, self.parallel_manager.sp_group)
        #     x = split_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal"))
        #     T = x.shape[1]
        #     x_mask_org = x_mask
        #     x_mask = split_sequence(
        #         x_mask, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal")
        #     )

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        spatial_index = index[1]
        if spatial_index is not None:
            x = x.index_select(1, spatial_index)

        # === blocks ===
        for i, (spatial_block, temporal_block) in enumerate(zip(self.spatial_blocks, self.temporal_blocks)):
            if cache:
                temporal_cache, spatial_cache = cache[i]
            else:
                temporal_cache = spatial_cache = (None, None)
            
            x = spatial_block(
                x, y, t_mlp, y_lens, 
                T=T, S=S,
                cache=spatial_cache, index=index)
            
            x = temporal_block(
                x, y, t_mlp, y_lens, 
                T=T, S=S,
                cache=temporal_cache, index=index)

        # if self.parallel_manager.sp_size > 1:
        #     x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        #     x = gather_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="up", pad=get_pad("temporal"))
        #     T, S = x.shape[1], x.shape[2]
        #     x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        #     x_mask = x_mask_org

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)

        if cache is not None:
            if self.input_cache:
                x = self.input_cache.update([x], spatial_index)[0]
                x = x.view(self.input_cache.bs, -1, self.input_cache.n_dim)
            else:
                assert index[0] is None, "The index should be None for now"

        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)

        # === Gather Output ===
        # if self.parallel_manager.cp_size > 1:
        #     x = gather_sequence(x, self.parallel_manager.cp_group, dim=0)

        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


class OpenSoraSTU(STUBase):
    def __init__(self, model: STDiT3C, filter_method="random_token"):
        super().__init__(model=model, filter_method=filter_method)
        self.cache = None
        self.patch_gather = None
        
    def init_generate_cache(self, x):
        B = x.size(0)
        assert B == 1, "The batch size should be 1 now"

        B = 2
        T, H, W = self.model.get_dynamic_size(x)
        S = H * W

        self.spat_size = S
        self.temp_size = T
        self.height = H
        self.width = W
        self.frame_update_counter = torch.zeros(T, dtype=torch.int16, device=x.device)

        dtype = self.model.x_embedder.proj.weight.dtype
        device = self.model.x_embedder.proj.weight.device

        last_temporal_block = self.model.temporal_blocks[-1]
        # last_temporal_block.init_cache(B, T, S)
        self.model.init_cache(B, T, S)

        # init cache
        self.cache = []
        for _ in range(self.model.depth):
            temporal_self_attn_cache = QKVCache(3, S, B, T * S, self.model.hidden_size, dtype=dtype, device=device)
            temporal_cross_attn_cache = QKVCache(1, T, B, T * S, self.model.hidden_size, dtype=dtype, device=device)
            spatial_self_attn_cache = QKVCache(3, T, B, T * S, self.model.hidden_size, dtype=dtype, device=device)
            spatial_cross_attn_cache = QKVCache(1, T, B, T * S, self.model.hidden_size, dtype=dtype, device=device)
            self.cache.append((
                (temporal_self_attn_cache, temporal_cross_attn_cache), 
                (spatial_self_attn_cache, spatial_cross_attn_cache)))
        
        # init patch gather
        self.patch_gather = PatchGather(patch_size=self.model.x_embedder.patch_size)
        self.patch_gather = self.patch_gather.to(dtype=torch.float32, device=x.device)

        # init dual filter counter
        self.filter_counter = 0

        # init short-term window
        if self.filter_method_int >= 3:
            self.window_length = 4
            self.window = ShorttermWindow(x, self.window_length)

    def generate_check(self, x):
        B = x.size(0)
        T, H, W = self.model.get_dynamic_size(x)
        S = H * W

        assert (2, self.temp_size, self.spat_size) == (B, T, S), "The shape of x is not the same as the cache shape"
    
    def to_bcthw(self, x):
        return x

    def get_patch_size(self):
        return self.patch_gather.patch_size
    
    def get_thw(self, x):
        return (x.size(2), x.size(3), x.size(4))

    @torch.no_grad()
    # ADD use cache parameter
    def forward(self, x, *args, **kwargs):
        use_cache = kwargs.get("use_cache", False)
        if use_cache and self.cache is not None:
            self.generate_check(x)
            kwargs["cache"] = self.cache
        
        return self.model(x, *args, **kwargs)
