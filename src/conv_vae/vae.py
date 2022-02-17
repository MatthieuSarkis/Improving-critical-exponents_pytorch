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
import time
import os
import torch
import torch.utils.data
from torch import nn
from torchvision.utils import save_image
from typing import Tuple

from src.conv_vae.encoder import Encoder
from src.conv_vae.decoder import Decoder
from src.conv_vae.loss_function import VAE_loss
from src.utils import plot_losses

class Conv_VAE(nn.Module):

    def __init__(
        self, 
        hidden_dim: int,
        latent_dim: int, 
        properties_dim: int, 
        network_name: str,
        lattice_size: int,
        kl_ratio: float = 1.0,
        reg_ratio: float = 1.0,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './saved_models',
    ) -> None:

        self.constructor_args = locals()
        del self.constructor_args['self']
        del self.constructor_args['__class__']

        super(Conv_VAE, self).__init__()

        self.network_name = network_name
        self.device = device
        self.save_dir = save_dir
        self.save_dir_model = os.path.join(self.save_dir, self.network_name+'_model')
        self.save_dir_ckpts = os.path.join(self.save_dir_model, 'ckpts')
        self.save_dir_images = os.path.join(self.save_dir, 'images')
        os.makedirs(self.save_dir_images, exist_ok=True)
        os.makedirs(self.save_dir_ckpts, exist_ok=True)

        self.encoder = Encoder(
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim, 
            properties_dim=properties_dim, 
            lattice_size=lattice_size
        )

        self.decoder = Decoder(
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim, 
            properties_dim=properties_dim, 
            lattice_size=lattice_size, 
            save_dir_images=self.save_dir_images, 
            device=self.device
        )

        self.criterion = VAE_loss(kl_ratio=kl_ratio, reg_ratio=reg_ratio)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)
    
    def forward(
        self, 
        x: torch.tensor, 
        p: torch.tensor,
    ) -> Tuple[torch.tensor, float, float]:

        x = x.to(self.device)
        p = p.to(self.device)

        mu, log_var = self.encoder.forward(x, p)
        z = self._reparameterize(mu, log_var)
        reconstructed = self.decoder.forward(z, p)

        return reconstructed, mu, log_var

    def check_reconstruction(
        self,
        inputs: torch.tensor,
        properties: torch.tensor,
        epoch: int,
    ) -> None:
    
        inputs = inputs.to(self.device)
        properties = properties.to(self.device)

        with torch.no_grad():

            outputs, _, _ = self(inputs, properties)

            inputs = 2 * inputs - 1
            outputs = torch.sign(2 * outputs - 1)

            comparison = torch.cat([inputs, outputs]).to('cpu')

            save_image(
                comparison, 
                os.path.join(self.save_dir_images, 'reconstruction_epoch={}.png'.format(epoch)), 
                nrow=5
            )

    @staticmethod
    def _reparameterize( 
        mu: float, 
        log_var: float,
    ) -> torch.tensor:

        std = torch.exp(0.5 * log_var)
        normal = torch.randn_like(std)

        return mu + std * normal

    def _train(
        self,
        epochs: int,
        X_train: torch.tensor,
        X_test: torch.tensor,
        y_train: torch.tensor,
        y_test: torch.tensor,
        batch_size: int,
        save_checkpoints: bool=False,
    ) -> None:
    
        X_train, X_test, y_train, y_test = X_train.to(self.device), X_test.to(self.device), y_train.to(self.device), y_test.to(self.device)
        X_train, X_test = (X_train + 1) / 2, (X_test + 1) / 2

        self.loss_history = {
            'train': [], 
            'test': [], 
            'train_reconstruction': [], 
            'test_reconstruction': [], 
            'train_kl': [], 
            'test_kl': []
        }
    
        for epoch in range(epochs):
            
            initial_time = time.time()
            
            self._train_one_epoch(X_train, y_train, batch_size)
            self._test_one_epoch(X_test, y_test, batch_size)

            if epoch % 5 == 0:

                self.check_reconstruction(
                    inputs=X_test[:5], 
                    properties=y_test[:5], 
                    epoch=epoch
                )

                self.decoder.sample_images(
                    n_images_per_p=8, 
                    properties=[0.5928], 
                    directory_path=self.save_dir_images, 
                    epoch=epoch
                )

            print("Epoch: {}/{}, Train Loss: {:.1f}, Test Loss: {:.1f}, Time: {:.2f}s".format(epoch+1, epochs, self.loss_history['train'][-1], self.loss_history['test'][-1], time.time()-initial_time))

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

            plot_losses(
                path_to_loss_history=os.path.join(self.save_dir, 'loss.json'),
                save_directory=os.path.join(self.save_dir, 'losses')
            )

        checkpoint_dict = {
            'constructor_args': self.constructor_args,
            'model_state_dict': self.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
        }

        torch.save(checkpoint_dict, os.path.join(self.save_dir_model, 'final_model.pt'))

    def _train_one_epoch(
        self,
        X_train: torch.tensor,
        y_train: torch.tensor,
        batch_size: int,
    ) -> None:
        r"""Performs one epoch of training"""

        permutation = torch.randperm(X_train.shape[0])
        train_loss = 0.0
        train_reconstruction_loss = 0.0 
        self.train()

        for i in range(0, X_train.shape[0], batch_size):

            indices = permutation[i:i+batch_size]
            inputs = X_train[indices].to(self.device)
            properties = y_train[indices].to(self.device)

            self.optimizer.zero_grad()
            outputs, mu, log_var = self(inputs, properties)

            total_loss, reconstruction_loss = self.criterion(outputs, inputs, mu, log_var)
            total_loss.backward()
            self.optimizer.step()
            train_loss += total_loss.item()
            train_reconstruction_loss += reconstruction_loss.item()

        train_loss /= X_train.shape[0] 
        train_reconstruction_loss /= X_train.shape[0]  
        train_kl_loss = train_loss - train_reconstruction_loss

        self.loss_history['train'].append(train_loss)
        self.loss_history['train_reconstruction'].append(train_reconstruction_loss)
        self.loss_history['train_kl'].append(train_kl_loss)

    def _test_one_epoch(
        self,
        X_test: torch.tensor,
        y_test: torch.tensor,
        batch_size: int,
    ) -> None:

        permutation = torch.randperm(X_test.shape[0])
        test_loss = 0.0
        test_reconstruction_loss = 0.0 
        self.eval()

        for i in range(0, X_test.shape[0], batch_size):

            indices = permutation[i:i+batch_size]
            inputs = X_test[indices].to(self.device)
            properties = y_test[indices].to(self.device)

            outputs, mu, log_var = self(inputs, properties)
            loss, reconstruction_loss = self.criterion(outputs, inputs, mu, log_var)
            test_loss += loss.item()
            test_reconstruction_loss += reconstruction_loss.item()

        test_loss /= X_test.shape[0]
        test_reconstruction_loss /= X_test.shape[0]
        test_kl_loss = test_loss - test_reconstruction_loss

        self.loss_history['test'].append(test_loss)
        self.loss_history['test_reconstruction'].append(test_reconstruction_loss)
        self.loss_history['test_kl'].append(test_kl_loss)