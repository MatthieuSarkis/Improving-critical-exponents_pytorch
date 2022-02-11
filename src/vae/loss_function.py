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

class VAE_loss(nn.Module):

    def __init__(
        self,
        kl_bce_ratio: float=0.5,
    ) -> None:

        super(VAE_loss, self).__init__()
        self.kl_bce_ratio = kl_bce_ratio

    def forward(
        self, 
        reconstructed: torch.tensor, 
        target: torch.tensor, 
        mu: float, 
        log_var: float,
    ) -> torch.tensor:

        reconstruction = F.binary_cross_entropy(reconstructed, target, reduction='sum')
        kld = self.kl_bce_ratio * torch.sum(log_var.exp() + mu.pow(2) - 1 - log_var)

        return reconstruction + kld