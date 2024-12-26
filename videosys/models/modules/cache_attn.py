from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from videosys.models.modules.normalization import LlamaRMSNorm


class QKVCache:
    def __init__(self, n_cache, t_frame, bs, seq_len, n_dim, dtype, device):
        self.n_cache = n_cache
        self.t_frame = t_frame
        self.bs = bs
        self.seq_len = seq_len
        self.n_tokens = seq_len // t_frame
        self.n_dim = n_dim

        self.cache = torch.empty(n_cache, bs, seq_len, n_dim, dtype=dtype, device=device)

    @property
    def shape(self):
        # The shape is supposed to be (B, T, S, C)
        return (self.bs, self.t_frame, self.n_tokens, self.n_dim)

    @property
    def packed(self):
        ret = [self.cache[i].view(self.shape) for i in range(self.n_cache)]
        return ret

    def update(self, values, token_index=None):
        # value: (B, N, C)
        if token_index is None:
            for i in range(self.n_cache):
                self.cache[i].copy_(values[i])
        else:
            for i in range(self.n_cache):
                self.cache[i].index_copy_(1, token_index, values[i])

        return self.packed


class OpenSoraAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def forward(
        self, 
        x: torch.Tensor, 
        T: int,
        S: int,
        cache: Optional[QKVCache] = None,
        token_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (S > B * T)
        partial_input_flag = (N < T * S)

        qkv = self.qkv(x)  # (B, N, 3 * C)
        if cache:  # update cache and get qkv
            if N == S * T:
                # if the input is full sequence, no index is needed
                assert token_index is None

            qkv = qkv.view(B, N, 3, C).permute(2, 0, 1, 3).unbind(0)  # q, k, v: (B, N, C)
            packed_qkv = cache.update(qkv, token_index)  # q, k, v: (B, T, S, C)

            packed = list()
            for v in packed_qkv:
                packed.append(v.view(B * T, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3))  # (B * T, H, S, D)
            q, k, v = packed
        else:
            qkv = rearrange(qkv, "b (t s) (c h d) -> c (b t) h s d", t=T, s=S, c=3, h=self.num_heads, d=self.head_dim)
            q, k, v = qkv.unbind(0)

        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = rearrange(x, "(b t) s h d -> b (t s) (h d)", t=T, s=S)

        # partial output
        if partial_input_flag and token_index is not None:
            x = x.index_select(1, token_index)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OpenSoraMultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, enable_flash_attn=False):
        super(OpenSoraMultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_flash_attn = enable_flash_attn

    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor, 
        T: int,
        S: int,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[QKVCache] = None,
        token_index: Optional[torch.Tensor] = None,
    ):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape 
        partial_input_flag = (N < T * S)

        q = self.q_linear(x)
        if cache:
            if N == S * T:
                # if the input is full sequence, no index is needed
                assert token_index is None
            q = cache.update((q,), token_index)[0]  # q: (B, S, T, C)
        
        q = q.view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        if self.enable_flash_attn:
            x = self.flash_attn_impl(q, k, v, mask, B, S * T, C)
        else:
            x = self.torch_impl(q, k, v, mask, B, S * T, C)

        if partial_input_flag and token_index is not None:
            x = x.index_select(1, token_index)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flash_attn_impl(self, q, k, v, mask, B, N, C):
        from flash_attn import flash_attn_varlen_func

        q_seqinfo = _SeqLenInfo.from_seqlens([N] * B)
        k_seqinfo = _SeqLenInfo.from_seqlens(mask)

        x = flash_attn_varlen_func(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_heads, self.head_dim),
            v.view(-1, self.num_heads, self.head_dim),
            cu_seqlens_q=q_seqinfo.seqstart.cuda(),
            cu_seqlens_k=k_seqinfo.seqstart.cuda(),
            max_seqlen_q=q_seqinfo.max_seqlen,
            max_seqlen_k=k_seqinfo.max_seqlen,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.view(B, N, C)
        return x

    def torch_impl(self, q, k, v, mask, B, N, C):
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = torch.zeros(B, 1, N, k.shape[2], dtype=torch.bool, device=q.device)
        for i, m in enumerate(mask):
            attn_mask[i, :, :, :m] = -1e9

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = out.transpose(1, 2).contiguous().view(B, N, C)
        return x


@dataclass
class _SeqLenInfo:
    """
    from xformers

    (Internal) Represents the division of a dimension into blocks.
    For example, to represents a dimension of length 7 divided into
    three blocks of lengths 2, 3 and 2, use `from_seqlength([2, 3, 2])`.
    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 2, 5, 7]
        seqstart: torch.IntTensor([0, 2, 5, 7])
    """

    seqstart: torch.Tensor
    max_seqlen: int
    min_seqlen: int
    seqstart_py: List[int]

    def to(self, device: torch.device) -> None:
        self.seqstart = self.seqstart.to(device, non_blocking=True)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        yield from zip(self.seqstart_py, self.seqstart_py[1:])

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> "_SeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        assert not isinstance(seqlens, torch.Tensor)
        seqstart_py = [0]
        max_seqlen = -1
        min_seqlen = -1
        for seqlen in seqlens:
            min_seqlen = min(min_seqlen, seqlen) if min_seqlen != -1 else seqlen
            max_seqlen = max(max_seqlen, seqlen)
            seqstart_py.append(seqstart_py[len(seqstart_py) - 1] + seqlen)
        seqstart = torch.tensor(seqstart_py, dtype=torch.int32)
        return cls(
            max_seqlen=max_seqlen,
            min_seqlen=min_seqlen,
            seqstart=seqstart,
            seqstart_py=seqstart_py,
        )
    

class PatchGather(nn.Module):
    """Add all std scale in one patch.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(1, 1, kernel_size=patch_size, stride=patch_size)

        self.proj.weight.requires_grad = False
        self.proj.bias.requires_grad = False
        nn.init.constant_(self.proj.weight, 1.0)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        
        return x
