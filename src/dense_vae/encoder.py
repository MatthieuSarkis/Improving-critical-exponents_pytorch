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
        lattice_size: int,
    ) -> None:

        super(Encoder, self).__init__()

        self.input_dim = lattice_size**2
        self.hidden_dim = hidden_dim
        self.properties_dim = properties_dim
        self.latent_dim = latent_dim

        self.fc  = nn.Linear(self.input_dim + properties_dim, self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.hidden_dim, latent_dim)

    def forward(
        self, 
        x: torch.tensor, 
        p: torch.tensor,
    ) -> Tuple[float, float]:

        h = torch.cat([x, p], dim=1)
        h = self.fc(h)
        h = F.relu_(h)
        z_mu = self.fc_mu(h)
        z_var = self.fc_log_var(h)

        return z_mu, z_var