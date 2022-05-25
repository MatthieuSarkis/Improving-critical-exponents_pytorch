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

def kl_ratio_schedule(
    epoch: int,
    epochs: int,
    min: float = 0.5,
    max: float = 1.0,
) -> float:

    return (max - min) * epoch / (epochs / 10) + min if epoch <= epochs // 10 else max

def plot_losses(
    path_to_loss_history: str,
    save_directory: str = './losses',
) -> None:

    with open(path_to_loss_history) as f:
        loss_history = json.load(f)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    for key, value in loss_history.items():
        name = re.split('_', key)[-1]
        if name == 'total':
            _ = axs[0][0].plot(value, label=key)
        elif name == 'reconstruction': 
            _ = axs[0][1].plot(value, label=key)
        elif name == 'kl':
            _ = axs[1][1].plot(value, label=key)
        elif name == 'regularization':
            _ = axs[1][0].plot(value, label=key)

    for ax in axs.flatten():
        ax.set_facecolor('lightblue')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('Epoch')

    #fig.patch.set_facecolor('xkcd:mint green')

    axs[0][0].set_title('Total Loss')
    axs[0][1].set_title('Reconstruction Loss')
    axs[1][1].set_title('KL divergence')
    axs[1][0].set_title('Regularization')
           
    plt.tight_layout()

    fig.savefig(save_directory)
    plt.close(fig)