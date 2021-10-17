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


from argparse import ArgumentParser
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from typing import Tuple, Dict

from network import generator
from utils import generator_loss
from logger import Logger

def main(args):

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    
    generator_model = generator(noise_dim=args.noise_dim, device=args.device)
    cnn_model = torch.load(args.CNN_model_path)   
    optimizer = optim.Adam(params=generator_model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    logger = Logger(save_dir=save_dir)

    generator_model, loss = train(epochs=args.epochs,
                                  logger=logger,
                                  batch_size=args.batch_size,
                                  noise_dim=args.noise_dim,
                                  generator_model=generator_model,
                                  cnn_model=cnn_model,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  ckpt_freq=args.ckpt_freq,
                                  bins_number=args.bins_number,
                                  device=args.device,
                                  set_generate_plots=args.set_generate_plots)

    logger.save_metadata(vars(args))


def train(epochs: int,
          logger: Logger,
          batch_size: int,
          noise_dim: int,
          generator_model: torch.nn.Module,
          cnn_model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.modules.loss._Loss,
          ckpt_freq: int,
          bins_number: int,
          device: str,
          set_generate_plots: bool = False,
          ) -> Tuple[nn.Module, Dict]:
    
    generator_model.train()
    cnn_model.eval()
    
    for epoch in range(epochs):

        logger.set_time_stamp(1)
        noise = torch.randn(batch_size, noise_dim).to(device)

        optimizer.zero_grad()
        generated_images = generator_model(noise)
        generated_images = torch.sign(generated_images)
        gen_loss = generator_loss(criterion, generated_images, cnn_model)
        gen_loss.backward()
        optimizer.step()

        logger.save_checkpoint(model=generator_model,
                               optimizer=optimizer,
                               epoch=epoch,
                               ckpt_freq=ckpt_freq)
            
        logger.set_time_stamp(2)
        logger.logs['loss'].append(gen_loss.item())
        logger.save_logs()
        
        if set_generate_plots:
            logger.generate_plots(generator=generator_model,
                                  cnn=cnn_model,
                                  epoch=epoch,
                                  noise_dim=noise_dim,
                                  bins_number=bins_number)
        logger.print_status(epoch=epoch)

    logger.save_checkpoint(model=generator_model, is_final_model=True)
    
    return generator_model, logger.logs
    
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bins_number", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=10e-3)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./saved_models/gan_cnn_regression")
    parser.add_argument("--ckpt_freq", type=int, default=10)
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/CNN_L128_N10000/saved-model.h5")
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument('--set_generate_plots', dest='set_generate_plots', action='store_true')
    parser.add_argument('--no-set_generate_plots', dest='set_generate_plots', action='store_false')
    parser.set_defaults(set_generate_plots=False)


    args = parser.parse_args()
    main(args)
