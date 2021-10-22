# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv

class ConvCell(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 ) -> None:
        
        super(ConvCell, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn = nn.BatchNorm2d(output_dim)
   
    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu_(x)
        return x

class Discriminator(nn.Module):
    
    def __init__(self,
                 lattice_size: int = 128,
                 n_conv_cells: int = 3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 ) -> None:
        
        super(Discriminator, self).__init__()
        
        self.n_conv_cells = n_conv_cells
        self.L = lattice_size
        self.device = device
        
        conv_block = [ConvCell(input_dim=1, output_dim=64)]
        for i in range(n_conv_cells):
            conv_block.append(ConvCell(input_dim=64*2**i, output_dim=64*2**(i+1)))
        self.conv_block = nn.Sequential(*conv_block)
        
        self.linear = nn.Linear(in_features=self._get_dimension(), out_features=1)
        self.bn = nn.BatchNorm1d(1)
  
        self.to(self.device)
  
    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = x.to(self.device)
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.bn(x)
        x = torch.sigmoid_(x)
        return x
        
    def _get_dimension(self) -> int:
        
        x = torch.zeros((1, 1, self.L, self.L))
        x = self.conv_block(x)
        return int(torch.numel(x))   
