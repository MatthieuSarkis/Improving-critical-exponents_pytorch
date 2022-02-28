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
from typing import Tuple

class VAE_loss(nn.Module):

    def __init__(
        self,
        reg_ratio: float = 1.0,
    ) -> None:

        super(VAE_loss, self).__init__()
        self.reg_ratio = reg_ratio

    def forward(
        self, 
        reconstructed: torch.tensor, 
        target: torch.tensor, 
        mu: float, 
        log_var: float,
        kl_ratio: float = 0.5,
    ) -> Tuple[torch.tensor, torch.tensor]:

        #reconstruction = F.binary_cross_entropy(reconstructed, target, reduction='sum')
        #reconstruction /= torch.numel(reconstructed[0])

        reconstruction_criterion = nn.MSELoss(reduction='sum')
        reconstruction = reconstruction_criterion(reconstructed, target)

        #reconstruction_criterion = nn.L1Loss(reduction='sum')
        #reconstruction = reconstruction_criterion(reconstructed, target)

        kld = torch.sum(log_var.exp() + mu.pow(2) - 1 - log_var)
        regularization = torch.sum(torch.full_like(reconstructed, 1, dtype=torch.float32) - torch.abs(2 * reconstructed - 1))
        regularization /= torch.numel(reconstructed[0])

        return reconstruction + kl_ratio * kld + self.reg_ratio * regularization, reconstruction, kld, regularization