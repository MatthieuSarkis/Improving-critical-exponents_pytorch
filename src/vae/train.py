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
from src.vae.vae import VAE

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
        p_list=None,
        split=True,
        save_dir=None
    )

    # Instanciating and training the model
    vae = VAE(
        lattice_size=args.lattice_size,
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim,
        properties_dim=args.properties_dim,
        kl_bce_ratio=0.5,
        network_name='VAE',
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=save_dir
    )

    vae._train(
        epochs=args.epochs,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    
    parser = ArgumentParser()

    parser.add_argument("--dataset_size", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, default="./saved_models/vae")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--properties_dim", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.set_defaults(save_checkpoints=True)

    args = parser.parse_args()
    main(args) 