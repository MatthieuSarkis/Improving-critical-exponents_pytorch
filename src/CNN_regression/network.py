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
from torch.nn.modules.conv import ConvTranspose2d

class cnn(nn.Module):

    def __init__(self,
                 lattice_size: int = 128, 
                 n_conv_layers: int = 4, 
                 n_dense_layers: int = 3, 
                 n_neurons: int = 512, 
                 dropout_rate: float = 0,
                 device: str = 'cpu',
                 ) -> None:

        super(cnn, self).__init__()
        
        self.L = lattice_size
        self.L_after_conv = self.L // (2**n_conv_layers) 
        self.n_conv_layers = n_conv_layers
        self.device = device

        conv_block = [nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same'), 
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(32),
                      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(32),
                      nn.MaxPool2d(2)]
        
        for l in range(1, n_conv_layers):
            n_features = 32*(2**l)
            conv_block.append(nn.Conv2d(n_features//2, n_features, kernel_size=3, stride=1, padding='same'))
            conv_block.append(nn.ReLU(inplace=True))
            conv_block.append(nn.BatchNorm2d(n_features))
            conv_block.append(nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding='same'))
            conv_block.append(nn.ReLU(inplace=True))
            conv_block.append(nn.BatchNorm2d(n_features))
            conv_block.append(nn.MaxPool2d(2))
            
        for network in conv_block:
            network.apply(self._initialize_weights)
            
        self.conv_block = nn.Sequential(*conv_block) 

        dense_block = [nn.Linear(32 * 2**(self.n_conv_layers-1) * self.L_after_conv**2, n_neurons),
                       nn.ReLU(inplace=True),
                       nn.BatchNorm1d(n_neurons),
                       nn.Dropout(dropout_rate)]
        
        for _ in range(1, n_dense_layers):
            dense_block.append(nn.Linear(n_neurons, n_neurons))
            dense_block.append(nn.ReLU(inplace=True))
            dense_block.append(nn.BatchNorm1d(n_neurons))
            dense_block.append(nn.Dropout(dropout_rate))
                
        dense_block.append(nn.Linear(n_neurons, 1))
        dense_block.append(nn.Sigmoid())
        
        for network in dense_block:
            network.apply(self._initialize_weights)
            
        self.dense_block = nn.Sequential(*dense_block)
        
        self.to(device)

    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:

        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_block(x)
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
   
 
class cnn_test(nn.Module):

    def __init__(self,
                 lattice_size: int = 128, 
                 n_neurons: int = 512, 
                 device: str = 'str',
                 ) -> None:

        super(cnn_test, self).__init__()
        self.L = lattice_size 
        self.device = device

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same') 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(64*(self.L//4)**2, n_neurons) 
        self.linear2 = nn.Linear(n_neurons, 1)             
                      
        for network in [self.conv1, self.conv2, self.linear1, self.linear2]:
            network.apply(self._initialize_weights)
        
        self.to(device)
        
    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
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
    