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
from datetime import datetime
import json
import os
import torch

from src.data import generate_data
from src.conv_vae.vae import Conv_VAE

def main(args):

    # Preparing the directory to save all the logs of the run
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Saving the command line arguments of the run for future reference
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Importing the data
    X_train, y_train, X_test, y_test = generate_data(
        dataset_size=args.dataset_size,
        lattice_size=args.lattice_size,
        p_list=None if args.use_property else [0.5928],
        split=True,
        save_dir=None
    )

    # Instanciating and training the model
    vae = Conv_VAE(
        lattice_size=args.lattice_size,
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim,
        properties_dim=y_train.shape[1] if args.use_property else None,
        kl_ratio=args.kl_ratio,
        reg_ratio=args.reg_ratio,
        network_name='Convolutional_VAE',
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=save_dir
    )

    vae._train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train if args.use_property else None,
        y_test=y_test if args.use_property else None
    )

if __name__ == "__main__":
    
    parser = ArgumentParser()

    parser.add_argument("--dataset_size", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, default="./saved_models/conv_vae")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kl_ratio", type=float, default=0.5)
    parser.add_argument("--reg_ratio", type=float, default=1.0)
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--use_property', dest='use_property', action='store_true')
    parser.add_argument('--no-use_property', dest='use_property', action='store_false')
    parser.set_defaults(use_property=False)
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.set_defaults(save_checkpoints=True)

    args = parser.parse_args()
    main(args) 