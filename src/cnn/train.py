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
from src.cnn.cnn import CNN

def main(args):

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    X_train, y_train, X_test, y_test = generate_percolation_data(
        dataset_size=args.dataset_size,
        lattice_size=args.lattice_size,
        p_list=None,
        split=True,
        save_dir=None
    )
    
    model = CNN(
        lattice_size=args.lattice_size,
        n_conv_cells=4,
        n_dense_cells=3,
        n_neurons=512,
        dropout_rate=0.0,
        learning_rate=1e-4,
        device=args.device,
        save_dir=save_dir
    )
    
    model._train(
        epochs=args.epochs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=args.batch_size,
        save_checkpoints=args.save_checkpoints
    )

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

if __name__ == '__main__':

    parser = ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, default='./saved_models/cnn')
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--dataset_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.set_defaults(save_checkpoints=True)
    
    args = parser.parse_args()
    main(args)