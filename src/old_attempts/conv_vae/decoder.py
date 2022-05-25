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

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
 
class Decoder(nn.Module):

    def __init__(
        self, 
        latent_dim: int,  
        lattice_size: int,
        save_dir_images: str,
        n_convt_cells: int = 1,
        properties_dim: int = None,
        embedding_dim_decoder: int = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> None:

        super(Decoder, self).__init__()

        self.lattice_size = lattice_size
        self.properties_dim = properties_dim
        self.latent_dim = latent_dim
        self.embedding_dim_decoder = embedding_dim_decoder

        self.input_dim = latent_dim

        if properties_dim is not None:
            if embedding_dim_decoder is not None:
                self.embedding = nn.Linear(properties_dim, embedding_dim_decoder)
                self.input_dim += embedding_dim_decoder
            else:
                self.input_dim += properties_dim

        self.initial_size = self.lattice_size // 2**n_convt_cells
        self.fc = nn.Linear(self.input_dim, 256 * self.initial_size * self.initial_size)
        self.bn = nn.BatchNorm1d(256 * self.initial_size * self.initial_size)

        convt_block = []
        for i in reversed(range(9 - (n_convt_cells - 1), 9)):
            convt_block.append(ConvTransposeCell(2**i, 2**(i-1)))
        convt_block.append(nn.ConvTranspose2d(in_channels=256//2**(n_convt_cells - 1), out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1))
        convt_block.append(nn.BatchNorm2d(1))
        self.convt_block = nn.Sequential(*convt_block)

        self.device = device
        self.save_dir_images = save_dir_images

    def forward(
        self, 
        z: torch.tensor, 
        p: torch.tensor = None,
    ) -> torch.tensor:

        if p is not None:
            if self.embedding_dim_decoder is not None:
                p = self.embedding(p)
            z = torch.cat([z, p], dim=1)

        h = self.fc(z)
        h = F.leaky_relu_(h)
        h = self.bn(h)
        h = h.view(-1, 256, self.initial_size, self.initial_size)
        h = self.convt_block(h)
        h = torch.sigmoid_(h)

        return h

    def sample_images(
        self,
        n_images_per_p: int,
        properties: list = None,
        directory_path: str='',
        epoch: int=None,
    ) -> None:

        l = len(properties) if properties is not None else 1

        with torch.no_grad():
    
            for i in range(l):

                p = torch.full(size=(n_images_per_p, self.properties_dim), fill_value=properties[i]).to(self.device) if self.properties_dim is not None else None
                z = torch.randn(n_images_per_p, self.latent_dim).to(self.device)

                generated = self(z, p)
                generated = torch.sign(2 * generated - 1)

            if epoch is not None:
                name = 'generated_epoch={}_p={:.2f}'.format(epoch, properties[i]) if properties is not None else 'generated_epoch={}'.format(epoch)
                save_image(generated, os.path.join(directory_path, name) + '.png')
                return

            else:
                generated_numpy = generated.numpy()
                name = 'generated_p={0:.2f}'.format(properties[i]) if properties is not None else 'generated'
                np.save(os.path.join(directory_path, name), generated_numpy)

            
class ConvTransposeCell(nn.Module):
    r"""
    W_out = 2 * W_in
    H_out = 2 * W_in
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        
        super(ConvTransposeCell, self).__init__()
        
        self.convt1 = nn.ConvTranspose2d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            output_padding=1, 
            dilation=1
        )

        self.bn1 = nn.BatchNorm2d(output_dim)

        self.convt2 = nn.ConvTranspose2d(
            in_channels=output_dim, 
            out_channels=output_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            output_padding=0, 
            dilation=1
        )

        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        
        x = self.convt1(x)
        x = F.leaky_relu_(x)
        x = self.bn1(x)
        x = self.convt2(x)
        x = F.leaky_relu_(x)
        x = self.bn2(x)
        return x