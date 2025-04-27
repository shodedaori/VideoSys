import pytest

import torch

from diffusers.models.attention_processor import Attention
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

    print(base_attn.spatial_norm)


if __name__ == "__main__":
    test_self_attn(torch.device("cuda"))
