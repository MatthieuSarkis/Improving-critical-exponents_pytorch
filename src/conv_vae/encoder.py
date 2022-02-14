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
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class Encoder(nn.Module):

    def __init__(
        self, 
        hidden_dim: int,
        latent_dim: int,
        properties_dim: int, 
    ) -> None:

        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.properties_dim = properties_dim
        self.latent_dim = latent_dim

        self.conv_cell = ConvCell(input_dim=1+properties_dim, output_dim=self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.hidden_dim, latent_dim)

    def forward(
        self, 
        x: torch.tensor, 
        p: torch.tensor,
    ) -> Tuple[float, float]:

        p = torch.full_like(x, p) # we suppose here p is just a scalar
        h = torch.cat([x, p], dim=1) # concatenate along the color channel
        h = self.conv_cell(h)
        h = torch.flatten(h, start_dim=1) # don't flatten along the batch dimension
        z_mu = self.fc_mu(h)
        z_var = self.fc_log_var(h)

        return z_mu, z_var


class ConvCell(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        
        super(ConvCell, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            dilation=1
        )

        self.bn1 = nn.BatchNorm2d(output_dim)

        self.conv2 = nn.Conv2d(
            in_channels=output_dim, 
            out_channels=output_dim, 
            kernel_size=3, 
            stride=1, 
            padding='same'
        )

        self.bn2 = nn.BatchNorm2d(output_dim)
   
    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu_(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu_(x)
        return x