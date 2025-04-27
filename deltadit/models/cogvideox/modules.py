from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_3d_rotary_pos_embed(
    embed_dim, crops_coords, grid_size, temporal_size, theta: int = 10000, use_real: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    start, stop = crops_coords
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2).float() / dim_t))
    grid_t = torch.from_numpy(grid_t).float()
    freqs_t = torch.einsum("n , f -> n f", grid_t, freqs_t)
    freqs_t = freqs_t.repeat_interleave(2, dim=-1)

    # Spatial frequencies for height and width
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))
    grid_h = torch.from_numpy(grid_h).float()
    grid_w = torch.from_numpy(grid_w).float()
    freqs_h = torch.einsum("n , f -> n f", grid_h, freqs_h)
    freqs_w = torch.einsum("n , f -> n f", grid_w, freqs_w)
    freqs_h = freqs_h.repeat_interleave(2, dim=-1)
    freqs_w = freqs_w.repeat_interleave(2, dim=-1)

    # Broadcast and concatenate tensors along specified dimension
    def broadcast(tensors, dim=-1):
        num_tensors = len(tensors)
        shape_lens = {len(t.shape) for t in tensors}
        assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
        shape_len = list(shape_lens)[0]
        dim = (dim + shape_len) if dim < 0 else dim
        dims = list(zip(*(list(t.shape) for t in tensors)))
        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all(
            [*(len(set(t[1])) <= 2 for t in expandable_dims)]
        ), "invalid dimensions for broadcastable concatenation"
        max_dims = [(t[0], max(t[1])) for t in expandable_dims]
        expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
        tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
        return torch.cat(tensors, dim=dim)

    freqs = broadcast((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)

    t, h, w, d = freqs.shape
    freqs = freqs.view(t * h * w, d)

    # Generate sine and cosine components
    sin = freqs.sin()
    cos = freqs.cos()

    if use_real:
        return cos, sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Use for example in Lumina
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Use for example in Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)
    

class CogVideoXDownsample3D(nn.Module):
    # Todo: Wait for paper relase.
    r"""
    A 3D Downsampling layer using in [CogVideoX]() by Tsinghua University & ZhipuAI

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `2`):
            Stride of the convolution.
        padding (`int`, defaults to `0`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape

            # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.conv(x)
        # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x
    

class CogVideoXUpsample3D(nn.Module):
    r"""
    A 3D Upsample layer using in CogVideoX by Tsinghua University & ZhipuAI # Todo: Wait for paper relase.

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `1`):
            Stride of the convolution.
        padding (`int`, defaults to `1`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                # split first frame
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

                x_first = F.interpolate(x_first, scale_factor=2.0)
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                inputs = torch.cat([x_first, x_rest], dim=2)
            elif inputs.shape[2] > 1:
                inputs = F.interpolate(inputs, scale_factor=2.0)
            else:
                inputs = inputs.squeeze(2)
                inputs = F.interpolate(inputs, scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            # only interpolate 2D
            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = F.interpolate(inputs, scale_factor=2.0)
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        return inputs


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        return embeds
    

class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x
