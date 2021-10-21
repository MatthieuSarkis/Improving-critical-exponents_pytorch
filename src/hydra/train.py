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

from argparse import ArgumentParser
import os
from datetime import datetime
import json
import torch
from torch._C import device

from src.cnn.network import CNN
from src.hydra.hydra import Hydra
from src.hydra.data import generate_data_torch

def main(args):
    
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    real_images, _ = generate_data_torch(args.dataset_size)

    cnn_checkpoint = torch.load(args.CNN_model_path)
    cnn_checkpoint['device'] = args.device
    cnn = CNN(**cnn_checkpoint['constructor_args'])   
    cnn.load_state_dict(cnn_checkpoint['model_state_dict'])
    
    model = Hydra(cnn=cnn,
                  lattice_size=args.lattice_size,
                  noise_dim=args.noise_dim,
                  learning_rate=args.learning_rate,
                  l1=args.regularization_strength,
                  l2=args.discriminator_cnn_ratio,
                  device=args.device,
                  wanted_p=args.wanted_p,
                  save_dir=args.save_dir)
    
    model._train(epochs=args.epochs,
                 batch_size=args.batch_size,
                 real_images=real_images,
                 device=args.device,
                 set_generate_plots=args.set_generate_plots,
                 bins_number=args.bins_number)
    
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--dataset_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bins_number", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=10e-3)
    parser.add_argument("--regularization_strength", type=float, default=1.0)
    parser.add_argument("--discriminator_cnn_ratio", type=float, default=1.0)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--wanted_p", type=float, default=0.5928)
    parser.add_argument("--save_dir", type=str, default="./saved_models/hydra")
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/cnn_regression/2021.10.17.18.29.07/model/final_model.pt")
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument('--set_generate_plots', dest='set_generate_plots', action='store_true')
    parser.add_argument('--no-set_generate_plots', dest='set_generate_plots', action='store_false')
    parser.set_defaults(set_generate_plots=False)

    args = parser.parse_args()
    main(args)
