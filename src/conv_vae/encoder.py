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
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class Encoder(nn.Module):

    def __init__(
        self, 
        latent_dim: int,
        lattice_size: int,
        n_conv_cells: int = 1,
        properties_dim: int = None,
        embedding_dim_encoder: int = None, 
    ) -> None:

        super(Encoder, self).__init__()

        self.properties_dim = properties_dim
        self.latent_dim = latent_dim
        self.lattice_size = lattice_size
        self.embedding_dim_encoder = embedding_dim_encoder

        self.input_dim = 1

        if properties_dim is not None:
            if embedding_dim_encoder is not None:
                self.embedding = nn.Linear(properties_dim, lattice_size**2 * embedding_dim_encoder)
                self.input_dim += embedding_dim_encoder
            else:
                self.input_dim += properties_dim
        
        conv_block = [ConvCell(input_dim=self.input_dim, output_dim=64)]
        for i in range(n_conv_cells - 1):
            conv_block.append(ConvCell(input_dim=64*2**i, output_dim=64*2**(i+1)))
        self.conv_block = nn.Sequential(*conv_block)
        
        self.fc_mu = nn.Linear(self._get_dimension(), latent_dim)
        self.fc_log_var = nn.Linear(self._get_dimension(), latent_dim)

        self._initialize_weights()

    def forward(
        self, 
        x: torch.tensor, 
        p: torch.tensor = None,
    ) -> Tuple[float, float]:

        if p is not None:
            if self.embedding_dim_encoder is not None:
                p = nn.Embedding(p)
                p = p.view(-1, self.embedding_dim_encoder, self.lattice_size, self.lattice_size)
            else:
                p = p.view(-1, 1, 1, 1).repeat(1, 1, self.lattice_size, self.lattice_size)
            x = torch.cat([x, p], dim=1) # concatenate along the color channel
            
        h = self.conv_block(x)
        h = torch.flatten(h, start_dim=1) # don't flatten along the batch dimension
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        #log_var = torch.clamp(log_var, min=-10.0, max=50.0) 
        #log_var = (lambda y: torch.tanh(1e-3 * y))(log_var) # to handle the blow up of the variance in a smooth way

        return mu, log_var

    def _get_dimension(self) -> int:
        
        x = torch.zeros((1, self.input_dim, self.lattice_size, self.lattice_size))
        x = self.conv_block(x)
        return int(torch.numel(x))  

    def _initialize_weights(
        self,
    ) -> None:
    
        nn.init.constant_(self.fc_mu.weight.data, 0.0)
        nn.init.constant_(self.fc_mu.bias.data, 0.0)
        nn.init.constant_(self.fc_log_var.weight.data, 0.0)
        nn.init.constant_(self.fc_log_var.bias.data, 0.0)

class ConvCell(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        r"""
        W_out = W_in // 2
        H_out = W_in // 2
        """

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