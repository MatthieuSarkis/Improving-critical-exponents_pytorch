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
import json
import time
import torch
import torch.nn as nn
from typing import Dict

from utils import (
    plot_cnn_histogram, 
    plot_losses,
)

class Logger():
    """A helper class to better handle the saving of outputs."""
    
    def __init__(self,
                 save_dir: str,
                 ) -> None:
        """Constructor method for the Logger class.
        
        Args:
            save_dir: path to the main checkpoint directory, in which the logs
                      and plots subdirectories are located
                                  
        Returns:
            no value
        """
        
        self.save_dir = save_dir
        self.save_dir_model = os.path.join(self.save_dir, "model")
        self.save_dir_ckpts = os.path.join(self.save_dir_model, "ckpts")
        self.save_dir_logs = os.path.join(self.save_dir, "logs")
        self.save_dir_plots = os.path.join(self.save_dir, "plots")

        os.makedirs(self.save_dir_ckpts, exist_ok=True)
        os.makedirs(self.save_dir_logs, exist_ok=True)
        os.makedirs(self.save_dir_plots, exist_ok=True)
        
        self.logs: dict = {}
        self.logs['loss'] = []
        
        self.time_stamp = [0, 0]
        self.initial_value_portfolio = None
            
    def set_time_stamp(self,
                       i: int,
                       ) -> None:
        """Method to keep track of time stamps for monitoring job progress"""
        
        self.time_stamp[i-1] = time.time()
                            
    def print_status(self,
                     epoch: int,
                     ) -> None:
        """Method to print on the status of the run on the standard output"""
        
        print('    - Episode: {:<13d} | Loss: {:<13.2f} | Duration in seconds: {:<13.2f}'.format(epoch, 
                                                                                              self.logs["loss"][-1], 
                                                                                              self.time_stamp[1]-self.time_stamp[0]))
    
    def save_logs(self) -> None:
        """Saves all the necessary logs to 'save_dir_logs' directory."""
        
        with open(os.path.join(self.save_dir, 'logs.json'), 'w') as f:
            json.dump(self.logs, f,  indent=4, separators=(',', ': '))
        
    def generate_plots(self,
                       generator: nn.Module,
                       cnn: nn.Module,
                       epoch: int,
                       noise_dim: int = 100,
                       bins_number: int = 100,
                       ) -> None:
        """Call a helper function to plot the generator loss and the histogram."""
            
        plot_losses(losses_history=self.logs, 
                    figure_file=os.path.join(self.save_dir_plots, "generatorLoss"))

        plot_cnn_histogram(generator=generator,
                           cnn=cnn,
                           epoch=epoch,
                           save_dir=self.save_dir_plots,
                           noise_dim=noise_dim,
                           bins_number=bins_number)
     
    def save_checkpoint(self,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer = None,
                        epoch: int = None,
                        ckpt_freq: int = None,
                        is_final_model: bool = False):
        
        if is_final_model:
            torch.save(model.state_dict(), os.path.join(self.save_dir_model, 'final_model.pt'))
        
        else:
            if (epoch + 1) % ckpt_freq == 0:
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': self.logs['loss'],
                    }
                torch.save(checkpoint_dict, os.path.join(self.save_dir_ckpts, 'ckpt_{}.pt'.format(epoch)))

    def save_metadata(self,
                      args: Dict,
                      ) -> None:
        """Method to save the command line arguments into a json file."""

        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(args, f,  indent=4, separators=(',', ': '))
        