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

from src.data_factory.percolation import generate_percolation_data
from src.conv_vae.vae import Conv_VAE
from src.utils import train_test_split

def main(args):

    # Preparing the directory to save all the logs of the run
    save_dir = os.path.join(args.save_dir, args.stat_phys_model, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
    
    # Saving the command line arguments of the run for future reference
    with open(os.path.join(save_dir, 'logs', 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if args.stat_phys_model == "percolation":

        X_train, y_train, X_test, y_test = generate_percolation_data(
            dataset_size=args.dataset_size,
            lattice_size=args.lattice_size,
            p_list=None if args.use_property else [0.5928],
            split=True,
            save_dir=None
        )

    elif args.stat_phys_model == "ising":

        with open('./data/ising/L={}/T=2.2257.bin'.format(args.lattice_size), 'rb') as f:
           X = torch.frombuffer(buffer=f.read(), dtype=torch.int8, offset=0).reshape(-1, args.lattice_size, args.lattice_size)[:args.dataset_size]
           y = torch.full(size=(X.shape[0], 1), fill_value=2.2257)
        X_train, y_train, X_test, y_test = train_test_split(X, y)

    vae = Conv_VAE(
        lattice_size=args.lattice_size,
        latent_dim=args.latent_dim, 
        properties_dim=y_train.shape[1] if args.use_property else None,
        embedding_dim_encoder=args.embedding_dim_encoder,
        embedding_dim_decoder=args.embedding_dim_decoder,
        n_conv_cells=args.n_conv_cells,
        n_convt_cells=args.n_convt_cells,
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
        y_test=y_test if args.use_property else None,
        save_checkpoints=args.save_checkpoints
    )

if __name__ == "__main__":
    
    parser = ArgumentParser()

    parser.add_argument("--stat_phys_model", type=str, default="percolation")
    parser.add_argument("--dataset_size", type=int, default=2048)
    parser.add_argument("--save_dir", type=str, default="./saved_models/conv_vae")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--reg_ratio", type=float, default=1.0)
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--n_conv_cells", type=int, default=1)
    parser.add_argument("--n_convt_cells", type=int, default=1)
    parser.add_argument("--embedding_dim_encoder", type=int)
    parser.add_argument("--embedding_dim_decoder", type=int)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--use_property', dest='use_property', action='store_true')
    parser.add_argument('--no-use_property', dest='use_property', action='store_false')
    parser.set_defaults(use_property=False)
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.set_defaults(save_checkpoints=True)

    args = parser.parse_args()
    main(args) 