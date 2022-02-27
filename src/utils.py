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

import json
import matplotlib.pyplot as plt
import re
import torch
from typing import Tuple

def train_test_split(
    X: torch.tensor,
    y: torch.tensor,
    validation_fraction: float = 0.2,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    
    dataset_size = y.shape[0]
    
    idx = torch.randperm(dataset_size)
    X = X[idx]
    y = y[idx]
    
    split_idx = int(dataset_size * (1 - validation_fraction))
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, y_train, X_test, y_test

def plot_losses(
    path_to_loss_history: str,
    save_directory: str = './losses',
) -> None:

    with open(path_to_loss_history) as f:
        loss_history = json.load(f)

    fig, (ax_left, ax_middle, ax_right) = plt.subplots(1, 3, sharex=True, figsize=(15, 4))

    for key, value in loss_history.items():
        name = re.split('_', key)[-1]
        if name == 'reconstruction': 
            _ = ax_middle.plot(value, label=key)
        elif name == 'kl':
            _ = ax_right.plot(value, label=key)
        else:
            _ = ax_left.plot(value, label=key)

    for ax in (ax_left, ax_middle, ax_right):
        ax.legend()
        ax.grid(True)
        ax_left.set_xlabel('Epoch')

    ax_left.set_title('Total Loss')
    ax_middle.set_title('Reconstruction Loss')
    ax_right.set_title('KL divergence')
           
    fig.savefig(save_directory)
    plt.close(fig)