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

from src.generator_cnn.logger import Logger
from src.generator_cnn.utils import MSELossRegularized
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

        #for module in self.modules():
        #    module.apply(self._initialize_weights)

    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = self.convt1(x)
        x = F.leaky_relu_(x)
        x = self.bn1(x)
        x = self.convt2(x)
        x = F.leaky_relu_(x)
        x = self.bn2(x)
        return x
    
    @staticmethod 
    def _initialize_weights(module: torch.nn.Module) -> None:
        if type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
        elif type(module) == nn.BatchNorm2d:
            nn.init.normal_(module.weight, mean=1, std=0.02)
            
class Generator(nn.Module):
    
    def __init__(self,
                 cnn: nn.Module,
                 noise_dim: int = 100,
                 learning_rate: float = 10e-4,
                 l: float = 1.0,
                 device: str = 'cpu',
                 wanted_p: float = 0.5928,
                 save_dir: str = './saved_models/gan_cnn_regression',
                 ) -> None:

        self.constructor_args = locals()
        del self.constructor_args['self']
        del self.constructor_args['__class__']
        
        super(Generator, self).__init__()
        
        self.cnn = cnn
        self.cnn.eval()
        self.learning_rate = learning_rate
        self.l = l
        self.wanted_p = wanted_p
        self.save_dir = save_dir
        self.noise_dim = noise_dim
        self.logger = Logger(save_dir=self.save_dir)
        
        self.linear = nn.Linear(noise_dim, 2*2*256)
        self.bn = nn.BatchNorm1d(2*2*256)
        convt_block = []
        for i in reversed(range(4, 9)):
            convt_block.append(ConvTransposeCell(2**i, 2**(i-1)))
        convt_block.append(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1))
        convt_block.append(nn.BatchNorm2d(1))
        self.convt_block = nn.Sequential(*convt_block)
         
        #for module in self.modules():
        #    if not isinstance(module, nn.Sequential) and not isinstance(module, ConvTransposeCell):
        #        module.apply(self._initialize_weights)
          
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        self.criterion = MSELossRegularized(loss_function=nn.MSELoss(), 
                                            #loss_function=nn.L1Loss(),
                                            cnn=self.cnn, l=self.l, wanted_output=self.wanted_p)
        self.device = device  
        self.to(self.device)
    
    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:
        
        x = x.to(self.device)
        x = self.linear(x)
        x = F.leaky_relu_(x)
        x = self.bn(x)
        x = x.view(-1, 256, 2, 2)
        x = self.convt_block(x)
        x = (lambda y: torch.tanh(2.0 * y))(x)
        return x

    def train(self,
              epochs: int,
              batch_size: int,
              bins_number: int,
              set_generate_plots: bool = False,
              ) -> None:
    
        self.logger.initialize()
    
        self.train()
        self.cnn.eval()
        
        for epoch in range(epochs):

            self.logger.set_time_stamp(1)
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)

            self.optimizer.zero_grad()
            generated_images = self(noise)
            gen_loss = self.criterion(generated_images)
            gen_loss.backward()
            #print(torch.max(self.linear.weight.grad))
            self.optimizer.step()

            self.logger.save_checkpoint(model=self, epoch=epoch)
            self.logger.set_time_stamp(2)
            self.logger.logs['loss'].append(gen_loss.item())
            self.logger.save_logs()
            
            if set_generate_plots:
                self.logger.generate_plots(generator=self,
                                           cnn=self.cnn,
                                           epoch=epoch,
                                           noise_dim=self.noise_dim,
                                           bins_number=bins_number)
                
            self.logger.print_status(epoch=epoch,
                                     epochs=epochs)

        self.logger.save_checkpoint(model=self, is_final_model=True)
        
    @staticmethod 
    def _initialize_weights(module: torch.nn.Module) -> None:
        if type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight)
        elif type(module) == nn.BatchNorm2d or type(module) == nn.BatchNorm1d:
            nn.init.normal_(module.weight, mean=1, std=0.02)
        elif type(module) == nn.ConvTranspose2d:
            nn.init.xavier_normal_(module.weight)
        