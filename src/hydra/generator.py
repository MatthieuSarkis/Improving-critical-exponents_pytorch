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

class ConvTransposeCell(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 ) -> None:
        
        super(ConvTransposeCell, self).__init__()
        
        self.convt1 = nn.ConvTranspose2d(in_channels=input_dim, 
                                         out_channels=output_dim, 
                                         kernel_size=3, 
                                         stride=2, 
                                         padding=1, 
                                         output_padding=1, 
                                         dilation=1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.convt2 = nn.ConvTranspose2d(in_channels=output_dim, 
                                         out_channels=output_dim, 
                                         kernel_size=3, 
                                         stride=1, 
                                         padding=1, 
                                         output_padding=0, 
                                         dilation=1)
        self.bn2 = nn.BatchNorm2d(output_dim) 

    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = x.to(self.device)
        x = self.convt1(x)
        x = F.leaky_relu_(x)
        x = self.bn1(x)
        x = self.convt2(x)
        x = F.leaky_relu_(x)
        x = self.bn2(x)
        return x
          
class Generator(nn.Module):
    
    def __init__(self,
                 noise_dim: int = 100,
                 device: str = 'cpu',
                 ) -> None:

        self.constructor_args = locals()
        del self.constructor_args['self']
        del self.constructor_args['__class__']
        
        super(Generator, self).__init__()
            
        self.linear = nn.Linear(noise_dim, 2*2*256)
        self.bn = nn.BatchNorm1d(2*2*256)
        convt_block = []
        for i in reversed(range(4, 9)):
            convt_block.append(ConvTransposeCell(2**i, 2**(i-1)))
        convt_block.append(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1))
        convt_block.append(nn.BatchNorm2d(1))
        self.convt_block = nn.Sequential(*convt_block)
                
        self.device = device  
        self.to(self.device)
    
    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = x.to(self.device)
        x = self.linear(x)
        x = F.leaky_relu_(x)
        x = self.bn(x)
        x = x.view(-1, 256, 2, 2)
        x = self.convt_block(x)
        x = (lambda y: torch.tanh(2.0 * y))(x)
        return x