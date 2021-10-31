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

from src.hydra.generator import Generator
from src.hydra.discriminator import Discriminator
from src.hydra.logger import Logger
from src.hydra.utils import MSELossRegularized

class Hydra():
    
    def __init__(self,
                 cnn: nn.Module,
                 lattice_size: int = 128,
                 noise_dim: int = 100,
                 n_conv_cells: int = 3,
                 n_convt_cells: int = 5,
                 generator_learning_rate: float = 10e-4,
                 discriminator_learning_rate: float = 10e-3,
                 l1: float = 1.0,
                 l2: float = 1.0,
                 l3: float = 1.0,
                 patience_generator: int = 2,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 wanted_p: float = 0.5928,
                 save_dir: str = './saved_models/hydra',
                 ) -> None:

        self.constructor_args = locals()
        del self.constructor_args['self']
        del self.constructor_args['__class__']
        
        super(Hydra, self).__init__()
        
        self.device = device
        self.noise_dim = noise_dim
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.l2 = l2
        self.l3 = l3
        self.patience_generator = patience_generator
        
        self.logger = Logger(save_dir=save_dir)
        self.generator = Generator(noise_dim=noise_dim, n_convt_cells=n_convt_cells, device=device)
        self.discriminator = Discriminator(lattice_size=lattice_size, n_conv_cells=n_conv_cells, device=device)
        self.cnn = cnn.to(self.device)
        self.cnn.eval()
        
        self.generator_optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=self.generator_learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.discriminator_learning_rate)
        self.cnn_criterion = MSELossRegularized(loss_function=nn.L1Loss(), cnn=self.cnn, l=l1, wanted_output=wanted_p)
        self.bce_criterion = nn.BCELoss()
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
    def _train(self,
               epochs: int,
               batch_size: int,
               real_images: torch.tensor,
               set_generate_plots: bool = False,
               bins_number: int = 100,
               ) -> None:
        
        self.logger.initialize()
        
        for epoch in range(epochs):
            
            self.logger.set_time_stamp(1)
            
            permutation = torch.randperm(real_images.shape[0])
            generator_loss = 0.0
            discriminator_loss = 0.0
            correct_predictions = 0
            total = 0

            for i in range(0, real_images.shape[0], batch_size):

                indices = permutation[i:i+batch_size]
                number_images = indices.shape[0]
                 
                # Training the discriminator
                self.generator.eval()
                self.discriminator.train()
                
                noise = torch.randn(number_images, self.noise_dim, device=self.device)
                fake_images_batch = self.generator(noise)
                real_images_batch = real_images[indices].to(self.device)
                
                fake_output = self.discriminator(fake_images_batch).view(-1,)
                real_output = self.discriminator(real_images_batch).view(-1,)
                
                fake_label = torch.full((fake_output.shape[0],), 0.0, dtype=torch.float, device=self.device)
                real_label = torch.full((real_output.shape[0],), 1.0, dtype=torch.float, device=self.device)
                
                self.discriminator_optimizer.zero_grad()
                
                fake_gan_loss = self.bce_criterion(fake_output, fake_label)                
                real_gan_loss = self.bce_criterion(real_output, real_label)
                disc_loss = 0.5 * (real_gan_loss + fake_gan_loss)
                
                disc_loss.backward()
                self.discriminator_optimizer.step()
                
                discriminator_loss += disc_loss.item()
                
                # Computing the accuracy of the discriminator
                fake_output[fake_output<0.5] = 0
                fake_output[fake_output>=0.5] = 1
                real_output[real_output<0.5] = 0
                real_output[real_output>=0.5] = 1
                correct_predictions += (fake_output==fake_label).sum().item()
                correct_predictions += (real_output==real_label).sum().item()
                total += 2*number_images # set of real images and set of fake images, hence the 2
            
                # Training the generator
                self.generator.train()
                self.discriminator.eval()
                
                for _ in range(self.patience_generator):
                    
                    noise = torch.randn(number_images, self.noise_dim, device=self.device)
                    fake_images_batch = self.generator(noise)
                    
                    fake_output = self.discriminator(fake_images_batch).view(-1,)
                    fake_label = torch.full((number_images,), 1.0, dtype=torch.float, device=self.device)
                    
                    self.generator_optimizer.zero_grad()
                    
                    bce_loss = self.bce_criterion(fake_output, fake_label)
                    cnn_loss = self.cnn_criterion(fake_images_batch)
                    gen_loss = self.l2 * bce_loss + self.l3 * cnn_loss
                    
                    gen_loss.backward()
                    self.generator_optimizer.step()
                    
                    generator_loss += gen_loss.item()

                generator_loss /= self.patience_generator
                
            discriminator_loss /= (real_images.shape[0]//batch_size)
            generator_loss /= (real_images.shape[0]//batch_size)
            discriminator_accuracy = 100 * correct_predictions / total
            
            self.logger.logs['discriminator_loss'].append(discriminator_loss)
            self.logger.logs['generator_loss'].append(generator_loss)
            self.logger.logs['discriminator_accuracy'].append(discriminator_accuracy)
            self.logger.save_logs()
            self.logger.save_checkpoint(model=self, epoch=epoch)
            self.logger.set_time_stamp(2)
            self.logger.print_status(epoch=epoch, epochs=epochs)
            
            if set_generate_plots:
                self.logger.generate_plots(generator=self.generator,
                                           cnn=self.cnn,
                                           epoch=epoch,
                                           noise_dim=self.noise_dim,
                                           bins_number=bins_number)
         
        self.logger.save_checkpoint(model=self, is_final_model=True)