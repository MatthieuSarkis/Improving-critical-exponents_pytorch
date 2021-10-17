# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi and Matthieu Sarkis, https://github.com/adelshb, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import torch
from typing import Dict
import matplotlib.pyplot as plt

def generator_loss(loss_function: torch.nn.modules.loss._Loss, 
                   generated_images: torch.tensor,
                   cnn: torch.nn.Module,
                   wanted_output: float = 0.5928,
                   device: str = 'cpu',
                   ) -> torch.tensor:
    
    predicted_output = cnn(generated_images) 
    wanted_output = torch.full(predicted_output.shape, wanted_output, dtype=torch.float32, device=device)
    
    return loss_function(wanted_output, predicted_output)

def plot_cnn_histogram(generator: torch.nn.Module,
                       cnn: torch.nn.Module,
                       epoch: int,
                       save_dir: str,
                       noise_dim: int = 100,
                       bins_number: int = 100,
                       ) -> None:
    
    generator.eval()
    cnn.eval()
    
    test_size = bins_number**2
    noise = torch.randn(test_size, noise_dim)
    
    images = generator(noise)
    images = torch.sign(images)

    y_pred = cnn(images)

    fig, ax = plt.subplots(1, 1)
    ax.hist(y_pred, bins=bins_number, color='g')
    ax.set_title("Distribution of the value of p for GAN generated critical configurations")
    ax.set_xlabel("Control parameter p")
    ax.set_ylabel("Fraction of configurations")
    ax.set_xlim(0, 1)
    
    path = os.path.join(save_dir, "histograms")
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, "generatedImages_epoch{}.png".format(epoch)))

def plot_losses(losses_history: Dict,
                figure_file: str,
                ) -> None:
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)
    ax.plot(losses_history["loss"], label='generator')
    ax.grid(True)
    ax.legend()
    ax.set_title("Generator Loss history")
    fig.savefig(figure_file)

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('./data/generated/image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()
