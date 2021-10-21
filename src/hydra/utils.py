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
import torch
from typing import Dict
import matplotlib.pyplot as plt

class MSELossRegularized(torch.nn.Module):
    
    def __init__(self,
                 loss_function: torch.nn.modules.loss._Loss,
                 cnn: torch.nn.Module,
                 wanted_output: float = 0.5928,
                 l: float = 0.5,
                 ) -> None:
        
        super(MSELossRegularized, self).__init__()
        
        self.wanted_output = wanted_output
        self.loss_function = loss_function
        self.cnn = cnn
        self.l = l
        self.cnn.eval()

    def forward(self,
                generated_images: torch.tensor,
                ) -> torch.tensor:
        
        return self._generator_loss(generated_images)
        
    def _generator_loss(self, 
                        generated_images: torch.tensor,
                        ) -> torch.tensor:
    
        predicted_output = self.cnn(generated_images)
        wanted_output_ = torch.full_like(predicted_output, self.wanted_output, dtype=torch.float32)

        regularization = torch.sum(torch.full_like(generated_images, 1, dtype=torch.float32) - torch.abs(generated_images))
        regularization *= self.l / torch.numel(generated_images)

        return self.loss_function(wanted_output_, predicted_output) + regularization


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
