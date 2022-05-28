import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from src.progan.layers import *

class Generator(nn.Module):
    
    def __init__(
        self,
        factors: List[int],
        noise_dim: int,
        in_channels: int,
        img_channels: int = 1
    ) -> None:

        super(Generator, self).__init__()

        self.factors = factors

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(noise_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) - 1):
            
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(
        self,
        alpha: float,
        upscaled: torch.tensor,
        generated: torch.tensor
    ) -> torch.tensor:

        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(
        self,
        x: torch.tensor,
        alpha: float,
        steps: int
    ) -> torch.tensor:

        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)