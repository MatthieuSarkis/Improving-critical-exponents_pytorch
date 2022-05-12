import torch
import torch.nn as nn

class WSConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        gain: int = 2
    ) -> None:

        super(WSConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.scale = (gain / (in_channels * kernel_size**2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:

        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    
    def __init__(self) -> None:

        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:

        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pixelnorm: bool=True
    ) -> None:

        super(ConvBlock, self).__init__()

        self.use_pixelnorm = use_pixelnorm

        self.conv1 = WSConv2d(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = WSConv2d(in_channels=out_channels, out_channels=out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:

        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pixelnorm else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pixelnorm else x

        return x