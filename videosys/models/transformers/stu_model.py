from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn


class STUBase(nn.Module):
    """Base class for STU (Selective Token Update) models."""
    def __init__(self, model: nn.Module, filter_method="random_token"):
        super().__init__()
        self.model = model

        # Init this in function init_generate_cache
        self.spat_size = None
        self.temp_size = None
        self.height = None
        self.width = None
        self.patch_gather = None
        self.frame_update_counter = None
        
        # init filter method
        if filter_method == "random_token":
            self.filer_method_int = 0
        elif filter_method == "random_frame":
            self.filer_method_int = 1
        elif filter_method == "random_channel":
            self.filer_method_int = 2
        elif filter_method == "shortterm":
            self.filer_method_int = 3
        else:
            raise NotImplementedError("This filter method is not implemented")

    def init_generate_cache(self, x, *args, **kwargs):
        """Initialize the cache for the model."""
        raise NotImplementedError("This method should be implemented in the subclass")

    def generate_check(self, x):
        """Check if the cache is generated."""
        raise NotImplementedError("This method should be implemented in the subclass")
    
    @torch.no_grad()
    def global_select(self, weight: Optional[torch.Tensor], ratio: int, method: str = "random"):
        """Select a subset of tokens based on the given ratio."""
        T, S = self.temp_size, self.spat_size
        # weight: [T * S]
        if weight is not None:
            assert len(weight.shape) == 1, "The weight should be 1D tensor"
            assert weight.size(0) == T * S, "The weight size should be T * S"

        if method == "random":
            # Randomly select n_tokens from the permutation
            n_tokens = math.ceil(T * S * ratio)
            spatial_index = torch.randperm(T * S, device=weight.device)[:n_tokens]
        elif method == "largest":
            # Select the largest n_tokens from the weight
            n_tokens = math.ceil(T * S * ratio)
            _, spatial_index = torch.topk(weight, n_tokens, largest=True)
        elif method == "per_frame_largest":
            # Select the largest n_tokens from each frame
            n_tokens_per_frame = math.ceil(S * ratio)
            _, spatial_index = torch.topk(weight, n_tokens_per_frame, largest=True)
            spatial_index = spatial_index.view(-1)
        else:
            raise NotImplementedError("This method is not implemented")
        
        pos_b = spatial_index % S
        pos_a = spatial_index // S
        temporal_index = pos_b * T + pos_a

        return (temporal_index, spatial_index)
    
    @torch.no_grad()
    def frame_level_select(self, weight: Optional[torch.Tensor], ratio: int, method: str = "random"):
        """Select a subset of tokens based on the given ratio."""
        T, S = self.temp_size, self.spat_size
        # weight: [T]
        if weight is not None:
            assert len(weight.shape) == 1, "The weight should be 1D tensor"
            assert weight.size(0) == T, "The weight size should be T"

        if method == "random":
            # Randomly select n_tokens from the permutation
            n_tokens = math.ceil(T * ratio)
            t_index = torch.randperm(T, device=weight.device)[:n_tokens]
        elif method == "largest":
            # Select the largest n_tokens from the weight
            n_tokens = math.ceil(T * ratio)
            _, t_index = torch.topk(weight, n_tokens, largest=True)
        elif method == "less_frequent":
            # Select the less frequent n_tokens from the weight
            n_tokens = math.ceil(T * ratio)
            _, t_index = torch.topk(self.frame_update_counter, n_tokens, largest=False)
        else:
            raise NotImplementedError("This method is not implemented")
        
        # update per frame update counter
        self.frame_update_counter.index_add_(0, t_index, torch.ones_like(t_index, dtype=self.frame_update_counter.dtype))

        i = torch.arange(S, device=weight.device)  # [S]
        spatial_index = i[:, None] + t_index * S  # [S, n_tokens]
        spatial_index = spatial_index.view(-1)

        pos_b = spatial_index % S
        pos_a = spatial_index // S
        temporal_index = pos_b * T + pos_a

        return (temporal_index, spatial_index)

    @torch.no_grad()
    def channel_level_select(self, weight: Optional[torch.Tensor], ratio: int, method: str = "random"):
        """Select a subset of tokens based on the given ratio."""
        T, S = self.temp_size, self.spat_size
        # weight: [S]
        if weight is not None:
            assert len(weight.shape) == 1, "The weight should be 1D tensor"
            assert weight.size(0) == S, "The weight size should be S"

        if method == "random":
            # Randomly select n_tokens from the permutation
            n_tokens = math.ceil(S * ratio)
            s_index = torch.randperm(S, device=weight.device)[:n_tokens]
        elif method == "largest":
            # Select the largest n_tokens from the weight
            n_tokens = math.ceil(S * ratio)
            _, s_index = torch.topk(weight, n_tokens, largest=True)
        else:
            raise NotImplementedError("This method is not implemented")

        j = torch.arange(T, device=weight.device) * S
        spatial_index = s_index[:, None] + j  # [n_tokens, T]   
        spatial_index = spatial_index.view(-1)

        pos_b = spatial_index % S
        pos_a = spatial_index // S
        temporal_index = pos_b * T + pos_a

        return (temporal_index, spatial_index)

    @torch.no_grad()
    def random_token_filter(self, x, ratio, sparse_flag=False, **kwargs):
        # x: [B, C, T, H, W]
        assert x.size(0) == 1, "The batch size should be 1 now"
        if not sparse_flag:  # dense update
            return (None, None)

        return self.global_select(None, ratio, method="random")
    
    @torch.no_grad()
    def random_frame_filter(self, x, ratio, sparse_flag=False, **kwargs):
        # x: [B, C, T, H, W]
        assert x.size(0) == 1, "The batch size should be 1 now"
        if not sparse_flag:
            return (None, None)

        return self.frame_level_select(None, ratio, method="random")
    
    @torch.no_grad()
    def random_channel_filter(self, x, ratio, sparse_flag=False, **kwargs):
        # x: [B, C, T, H, W]
        assert x.size(0) == 1, "The batch size should be 1 now"
        if not sparse_flag:
            return (None, None)

        return self.channel_level_select(None, ratio, method="random")   
    
    @torch.no_grad()
    def shortterm_filter(self, x, ratio, sparse_flag=False, **kwargs):
        # x: [B, C, To, Ho, Wo]
        B, T, S = self.shape
        assert x.size(0) == 1, "The batch size should be 1 now"

        # update the short-term window
        self.window.insert(x)

        if not sparse_flag or self.filter_counter >= self.window_length - 1:
            # reset the counter, do not filter
            self.filter_counter = 0
            return (None, None)
        
        self.filter_counter += 1

        y = self.window.get_std_sqr()  # [B, C, To, Ho, Wo]

        if self.filter_counter == 3:
            y = torch.sqrt(y)
            prev_y = y[:, :, :-1, :, :]  # [B, C, To-1, Ho, Wo]
            next_y = y[:, :, 1:, :, :]   # [B, C, To-1, Ho, Wo]
            y = (torch.norm(next_y - prev_y, dim=1) ** 2).unsqueeze(1)  # [B, 1, To-1, Ho, Wo]
            
            # y_mean = torch.mean(y, dim=1, keepdim=True)  # [B, 1, To, Ho, Wo]
            # y = torch.norm(y - y_mean, dim=1, keepdim=True) ** 2  # [B, 1, To, Ho, Wo]
            
            y = self.patch_gather(y, flatten_flag=False).squeeze(1)  # [B, 1, T, H, W]
            y = torch.mean(y, dim=1).view(-1) # [S]

            return self.channel_level_select(y, ratio, method="largest")
        
        elif self.filter_counter == 4:
            y = torch.sum(y, dim=1, keepdim=True)  # [B, 1, To, Ho, Wo]
            y = self.patch_gather(y).view(-1)  # [N]

            return self.global_select(y, ratio, method="largest")
            
        else:
            y = torch.sum(y, dim=1, keepdim=True)  # [B, 1, To, Ho, Wo]
            y = self.patch_gather(y).view(T, S)  # [T, S]
            y = torch.sum(y, dim=-1)  # [T]
            
            return self.frame_level_select(y, ratio, method="largest")
        
    def to_bcthw(self, x) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def get_patch_size(self) -> Tuple[int, int, int]:
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def index_filter(self, v_pred, coef, **kwargs):
        if self.filer_method_int == 0:
            return self.random_token_filter(v_pred, coef, **kwargs)
        elif self.filer_method_int == 1:
            return self.random_frame_filter(v_pred, coef, **kwargs)
        elif self.filer_method_int == 2:
            return self.random_channel_filter(v_pred, coef, **kwargs)
        elif self.filer_method_int == 3:
            return self.shortterm_filter(v_pred, coef, **kwargs)
        else:
            raise NotImplementedError("This filter method is not implemented")
    
    @torch.no_grad()
    def get_update_mask(self, z, spatial_index=None):
        B = z.size(0)
        assert B == 1, "The batch size should be 1 now"
        
        _, _, T_n, H_n, W_n = z.size()
        if spatial_index is None:
            mask = torch.ones((B, 1, T_n, H_n, W_n), dtype=torch.int32)
            return mask

        _, T, S = self.shape
        H, W = self.height, self.width
        T_p, H_p, W_p = self.get_patch_size()
        mask = torch.zeros(B, T * S, dtype=torch.int32, device=z.device)
        mask.index_fill_(1, spatial_index, 1)

        mask = mask.view(B, T, 1, H, 1, W, 1)
        mask = mask.expand(B, T, T_p, H, H_p, W, W_p)
        mask = mask.reshape(B, T * T_p, H * H_p, W * W_p)
        mask = mask[:, :T_n, :H_n, :W_n]
        mask = mask.reshape(B, 1, T_n, H_n, W_n)
        return mask

    @torch.no_grad()
    # ADD use cache parameter
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in the subclass")
