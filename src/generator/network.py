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

class generator(nn.Module):
    
    def __init__(self,
                 noise_dim: int = 100,
                 device: str = 'cpu',
                 ) -> None:
        
        super(generator, self).__init__()
        
        self.device = device
        self.noise_dim = noise_dim
        
        self.linear = nn.Linear(noise_dim, 2*2*256)
        self.bn = nn.BatchNorm1d(2*2*256)
        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.convt3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.convt4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.convt5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.convt6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.convt7 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.convt8 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.convt9 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.bn9 = nn.BatchNorm2d(8)
        self.convt10 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, output_padding=0, dilation=1)
        self.bn10 = nn.BatchNorm2d(8)
        self.convt11 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
    
        for net in self.modules():
            net.apply(self._initialize_weights)
            
        self.to(device)
    
    def forward(self,
                x: torch.tensor):
        
        x = F.leaky_relu(self.bn(self.linear(x)), inplace=True)
        x = x.view(-1, 256, 2, 2)
        x = F.leaky_relu(self.bn1(self.convt1(x)), inplace=True)
        x = F.leaky_relu(self.bn2(self.convt2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.convt3(x)), inplace=True)
        x = F.leaky_relu(self.bn4(self.convt4(x)), inplace=True)
        x = F.leaky_relu(self.bn5(self.convt5(x)), inplace=True)
        x = F.leaky_relu(self.bn6(self.convt6(x)), inplace=True)
        x = F.leaky_relu(self.bn7(self.convt7(x)), inplace=True)
        x = F.leaky_relu(self.bn8(self.convt8(x)), inplace=True)
        x = F.leaky_relu(self.bn9(self.convt9(x)), inplace=True)
        x = F.leaky_relu(self.bn10(self.convt10(x)), inplace=True)
        x = torch.tanh(self.convt11(x))
        return x

    @staticmethod 
    def _initialize_weights(net: torch.nn.Module) -> None:
        """Xavier initialization of the weights in the Linear and Convolutional layers of a torch.nn.Module object.

        Args:
            net (torch.nn.Module): neural net whose weights are to be initialized

        Returns:
            no value
        """

        if type(net) == nn.Linear or type(net) == nn.Conv2d or type(net) == nn.ConvTranspose2d:
            nn.init.xavier_uniform_(net.weight)
            net.bias.data.fill_(1e-2)
