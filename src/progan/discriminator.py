import torch
import torch.nn as nn

from src.progan.config import FACTORS
from src.progan.layers import *
class Discriminator(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_channels: int = 1
    ) -> None:

        super().__init__()
        
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(FACTORS) - 1, 0, -1):

            conv_in_c = int(in_channels * FACTORS[i])
            conv_out_c = int(in_channels * FACTORS[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def fade_in(
        self,
        alpha: float,
        downscaled: torch.tensor,
        out: torch.tensor
    ) -> torch.tensor:
        
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1) # 512 -> 513

    def forward(
        self,
        x: torch.tensor,
        alpha: float,
        steps: int
    ) -> torch.tensor:

        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)