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

from src.generator.logger import Logger

class ConvTransposeCell(nn.Module):
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 ) -> None:
        
        super(ConvTransposeCell, self).__init__()
        
        self.convt1 = nn.ConvTranspose2d(in_channels=input_dim, 
                                         out_channels=output_dim, 
                                         kernel_size=3, 
                                         stride=2, 
                                         padding=1, 
                                         output_padding=1, 
                                         dilation=1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.convt2 = nn.ConvTranspose2d(in_channels=output_dim, 
                                         out_channels=output_dim, 
                                         kernel_size=3, 
                                         stride=1, 
                                         padding=1, 
                                         output_padding=0, 
                                         dilation=1)
        self.bn2 = nn.BatchNorm2d(output_dim) 

    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = self.convt1(x)
        x = self.bn1(x)
        x = F.leaky_relu_(x)
        x = self.convt2(x)
        x = self.bn2(x)
        x = F.leaky_relu_(x)
        return x
class Generator(nn.Module):
    
    def __init__(self,
                 noise_dim: int = 100,
                 learning_rate: float = 10e-3,
                 device: str = 'cpu',
                 save_dir: str = './saved_models/gan_cnn_regression',
                 ) -> None:
        
        super(Generator, self).__init__()
        
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.noise_dim = noise_dim
        self.logger = Logger(save_dir=self.save_dir)
        
        self.linear = nn.Linear(noise_dim, 2*2*256)
        self.bn = nn.BatchNorm1d(2*2*256)
        
        convt_block = []
        for i in reversed(range(4, 9)):
            convt_block.append(ConvTransposeCell(2**i, 2**(i-1)))
        convt_block.append(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1))
        self.convt_block = nn.Sequential(*convt_block)
    
        for module in self.modules():
            if not isinstance(module, nn.Sequential):
                module.apply(self._initialize_weights)
          
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.device = device  
        self.to(self.device)
    
    def forward(self,
                x: torch.tensor):
        
        x = x.to(self.device)
        x = F.leaky_relu(self.bn(self.linear(x)), inplace=True)
        x = x.view(-1, 256, 2, 2)
        x = self.convt_block(x)
        x = torch.tanh(x)
        return x

    @staticmethod 
    def _initialize_weights(module: torch.nn.Module) -> None:
        """Xavier initialization of the weights in the Linear and Convolutional layers of a torch.nn.Module object.

        Args:
            module (torch.nn.Module): module whose weights are to be initialized
        """

        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(1e-2)

    def _train(self,
               epochs: int,
               batch_size: int,
               cnn_model: torch.nn.Module,
               ckpt_freq: int,
               bins_number: int,
               set_generate_plots: bool = False,
               l: float = 0.5,
               ) -> None:
    
        self.logger.initialize()
    
        self.train()
        cnn_model.eval()
        
        for epoch in range(epochs):

            self.logger.set_time_stamp(1)
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)

            self.optimizer.zero_grad()
            generated_images = self(noise)
            generated_images = torch.sign(generated_images)
            gen_loss = self._generator_loss(loss_function=self.criterion, 
                                            generated_images=generated_images, 
                                            cnn=cnn_model, 
                                            device=self.device,
                                            wanted_output=0.5928,
                                            l=l)
            gen_loss.backward()
            self.optimizer.step()

            self.logger.save_checkpoint(model=self,
                                        optimizer=self.optimizer,
                                        epoch=epoch,
                                        ckpt_freq=ckpt_freq)
                
            self.logger.set_time_stamp(2)
            self.logger.logs['loss'].append(gen_loss.item())
            self.logger.save_logs()
            
            if set_generate_plots:
                self.logger.generate_plots(generator=self,
                                           cnn=cnn_model,
                                           epoch=epoch,
                                           noise_dim=self.noise_dim,
                                           bins_number=bins_number)
            self.logger.print_status(epoch=epoch)

        self.logger.save_checkpoint(model=self, is_final_model=True)
        
    @staticmethod 
    def _generator_loss(loss_function: torch.nn.modules.loss._Loss, 
                        generated_images: torch.tensor,
                        cnn: torch.nn.Module,
                        device: str = 'cpu',
                        wanted_output: float = 0.5928,
                        l: float = 0.5,
                        ) -> torch.tensor:
    
        predicted_output = cnn(generated_images).to(device) 
        wanted_output_ = torch.full_like(predicted_output, wanted_output, dtype=torch.float32, device=device)

        regularization = torch.sum(torch.full_like(generated_images, 1, dtype=torch.float32, device=device) - torch.abs(generated_images))
        regularization *= l / torch.numel(generated_images)

        return loss_function(wanted_output_, predicted_output) + regularization