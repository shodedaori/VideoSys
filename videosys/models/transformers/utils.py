import torch
import torch.nn as nn


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
        self.proj = nn.Conv3d(1, 1, kernel_size=patch_size, stride=patch_size, bias=False)

        self.proj.weight.requires_grad = False
        nn.init.constant_(self.proj.weight, 1.0)

    def forward(self, x, flatten_flag=True):
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
        
        if flatten_flag:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        
        return x



class ShorttermWindow:
    def __init__(self, z: torch.Tensor, l: int = 4):
        self.shape = z.shape
        self.length = l
        self.next_id = 0
        
        self.power2_sum = torch.zeros_like(z, dtype=torch.float32)
        self.power1_sum = torch.zeros_like(z, dtype=torch.float32)

        self.bucket = []
        for i in range(l):
            self.bucket.append(torch.zeros_like(z, dtype=torch.float32))

    def insert(self, z: torch.Tensor):
        # subtract the oldest value
        b = self.bucket[self.next_id]
        self.power2_sum -= b ** 2
        self.power1_sum -= b

        # update this bucket value
        b.copy_(z)

        # update the sum of squares and sum
        self.power2_sum += b ** 2
        self.power1_sum += b

        #  update the next_id
        self.next_id = (self.next_id + 1) % self.length

    def get_std_sqr(self):
        return (self.power2_sum - self.power1_sum ** 2 / self.length) / self.length
