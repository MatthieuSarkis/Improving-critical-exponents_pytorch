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
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import plot_losses

class ConvCell(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        
        super(ConvCell, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.max_pool = nn.MaxPool2d(2)
   
    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu_(x)
        x = self.bn2(x)
        x = self.max_pool(x)
        return x

class LinearCell(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: int = 0.0,
    ) -> None:
        
        super(LinearCell, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.tensor,
    ) -> torch.tensor:
        
        x = self.linear(x)
        x = F.relu_(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
      
class CNN(nn.Module):

    def __init__(
        self,
        lattice_size: int = 128, 
        n_conv_cells: int = 4, 
        n_dense_cells: int = 3, 
        n_neurons: int = 512, 
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './saved_models/cnn_regression',
    ) -> None:

        self.constructor_args = locals()
        del self.constructor_args['self']
        del self.constructor_args['__class__']

        super(CNN, self).__init__()
        
        self.device = device
        self.learning_rate = learning_rate
        self.L = lattice_size
        self.save_dir = save_dir
        self.loss_history = None
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        self.save_dir_ckpts = os.path.join(self.save_dir_model, 'ckpts')
        os.makedirs(self.save_dir_ckpts, exist_ok=True)

        conv_block = [ConvCell(1, 32)]        
        for l in range(1, n_conv_cells):
            conv_block.append(ConvCell(32*(2**(l-1)), 32*(2**l))) 
        self.conv_block = nn.Sequential(*conv_block) 
        
        dense_block = [LinearCell(self._get_dimension(), n_neurons, dropout_rate)]        
        for _ in range(1, n_dense_cells):
            dense_block.append(LinearCell(n_neurons, n_neurons, dropout_rate))        
        dense_block.append(nn.Linear(n_neurons, 1))
        dense_block.append(nn.Sigmoid())
        self.dense_block = nn.Sequential(*dense_block)
        
        #for module in self.modules():
        #    if not isinstance(module, nn.Sequential):
        #        module.apply(self._initialize_weights)
        
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='sum')
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5, verbose=False)
        self.to(self.device)
        
    def _get_dimension(self) -> int:
        r"""Helper method to read the shape of the data after the convolution blocks."""
        
        x = torch.zeros((1, 1, self.L, self.L))
        x = self.conv_block(x)
        return int(torch.numel(x))

    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:

        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_block(x)
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
   
    def _train_one_epoch(
        self,
        X_train: torch.tensor,
        y_train: torch.tensor,
        batch_size: int,
    ) -> None:

        permutation = torch.randperm(X_train.shape[0])
        train_loss = 0.0
        
        self.train()
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            inputs = X_train[indices].to(self.device)
            labels = y_train[indices].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= X_train.shape[0]
        self.loss_history['train'].append(train_loss)

    def _test_one_epoch(
        self,
        X_test: torch.tensor,
        y_test: torch.tensor,
        batch_size: int,
    ) -> None:

        permutation = torch.randperm(X_test.shape[0])
        test_loss = 0.0
        
        self.eval()

        for i in range(0, X_test.shape[0], batch_size):
            
            indices = permutation[i:i+batch_size]
            inputs = X_test[indices].to(self.device)
            labels = y_test[indices].to(self.device)
            outputs = self(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()
    
        test_loss /= X_test.shape[0]
        self.loss_history['test'].append(test_loss)

    def _train(
        self,
        epochs: int,
        X_train: torch.tensor,
        y_train: torch.tensor,
        X_test: torch.tensor,
        y_test: torch.tensor,
        batch_size: int,
        save_checkpoints: bool,
    ) -> None:
    
        self.loss_history = {'train': [], 'test': []}
    
        for epoch in range(epochs):
            
            initial_time = time.time()

            self._train_one_epoch(X_train=X_train, y_train=y_train, batch_size=batch_size)
            self._test_one_epoch(X_test=X_test, y_test=y_test, batch_size=batch_size)
            self.scheduler.step(self.loss_history['test'][-1])
            
            print("Epoch: {}/{}, Train Loss: {:.6f}, Test Loss: {:.6f}, Time: {:.2f}s".format(
                epoch+1, epochs, self.loss_history['train'][-1], self.loss_history['test'][-1], time.time()-initial_time))

            if save_checkpoints:
                if ((epoch+1) % 10) == 0:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'constructor_args': self.constructor_args,
                        'optimizer_state_dict': self.state_dict(),
                        'train_loss': self.loss_history['train'][-1],
                        'test_loss': self.loss_history['test'][-1],
                    }
                    torch.save(checkpoint_dict, os.path.join(self.save_dir_ckpts, 'ckpt_{}.pt'.format(epoch)))

            with open(os.path.join(self.save_dir, 'loss.json'), 'w') as f:
                json.dump(self.loss_history, f, indent=4)

            #plot_losses(
            #    path_to_loss_history=os.path.join(self.save_dir, 'loss.json'),
            #    save_directory=os.path.join(self.save_dir, 'losses')
            #)

        checkpoint_dict = {
            'constructor_args': self.constructor_args,
            'model_state_dict': self.state_dict(),
        }
        torch.save(checkpoint_dict, os.path.join(self.save_dir_model, 'final_model.pt'))