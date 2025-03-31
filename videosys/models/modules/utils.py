import torch
import torch.nn as nn


class Gaussian3d(nn.Module):
    def __init__(self, channel, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel = channel
        self.kernel_size = kernel_size
        self.sigma = kernel_size / 6.0

        self.conv = nn.Conv3d(channel, channel, kernel_size, padding='same', groups=channel, bias=False)
        self.conv.weight.requires_grad = False

        kernel = torch.arange(-(kernel_size // 2), (kernel_size + 1) // 2, dtype=torch.float32) ** 2
        kernel3d = torch.zeros((kernel_size, kernel_size, kernel_size), dtype=torch.float32)
        kernel3d += kernel.view(1, 1, -1)
        kernel3d += kernel.view(1, -1, 1)
        kernel3d += kernel.view(-1, 1, 1)

        kernel3d = torch.exp(-kernel3d / (2 * self.sigma ** 2))
        kernel3d /= kernel3d.sum()
        self.conv.weight.copy_(kernel3d.expand(channel, 1, -1, -1, -1))

    def forward(self, x):
        # x: [B, C, T, H, W]
        return self.conv(x)
