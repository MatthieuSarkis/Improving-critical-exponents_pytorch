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

class Discriminator(nn.Module):
    
    def __init__(self,
                 lattice_size: int = 128,
                 device: str = 'cpu',
                 ) -> None:
        
        super(Discriminator, self).__init__()
        
        self.L = lattice_size
        self.device = device
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.linear = nn.Linear(in_features=self._get_dimension(), out_features=1)
        self.bn3 = nn.BatchNorm1d(1)
    
        self.to(device)
    
    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = self.conv1(x)
        x = F.leaky_relu_(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.leaky_relu_(x)
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.bn3(x)
        x = torch.sigmoid_(x)
        
    def _get_dimension(self) -> int:
        
        x = torch.zeros((1, 1, self.L, self.L))
        x = self.conv1(x)
        x = self.conv2(x)
        return int(torch.numel(x))   
