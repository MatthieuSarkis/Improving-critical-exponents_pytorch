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
from torchvision.utils import save_image
 
class Decoder(nn.Module):

    def __init__(
        self, 
        hidden_dim: int,
        latent_dim: int, 
        properties_dim: int, 
        lattice_size: int,
        save_dir_images,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> None:

        super(Decoder, self).__init__()

        self.lattice_size = lattice_size
        self.input_dim = lattice_size**2
        self.hidden_dim = hidden_dim
        self.properties_dim = properties_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim + properties_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_dim)
    
        self.device = device
        self.save_dir_images = save_dir_images

    def forward(
        self, 
        z: torch.tensor, 
        p: torch.tensor,
    ) -> torch.tensor:

        h = torch.cat([z, p], dim=1)
        h = self.fc1(h)
        h = F.relu_(h)
        h = self.fc2(h)
        h = torch.sigmoid_(h)

        return h

    def sample_images(
        self,
        n_images: int,
        file_name: str='',
    ) -> None:

        with torch.no_grad():

            p = torch.full(size=(n_images, self.properties_dim), fill_value=0.5928).to(self.device)
            z = torch.randn(n_images, self.latent_dim).to(self.device)
            generated = self(z, p)
            generated = torch.sign(2 * generated - 1)

            save_image(generated.view(n_images, 1, self.lattice_size, self.lattice_size),
                       self.save_dir_images + '/samples_' + file_name + '.png')