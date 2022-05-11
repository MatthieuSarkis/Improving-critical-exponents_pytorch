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

import matplotlib.pyplot as plt
import torch
from typing import Tuple, Dict

def plot_losses(
    losses_history: Dict,
    figure_file: str,
) -> None:
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)
    ax.plot(losses_history["loss"], label='generator')
    ax.grid(True)
    ax.legend()
    ax.set_title("Generator Loss history")
    fig.savefig(figure_file)

def train_test_split(
    X: torch.tensor,
    y: torch.tensor,
    validation_fraction: float = 0.25,
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