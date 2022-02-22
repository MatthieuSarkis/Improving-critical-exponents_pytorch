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

from src.cnn.cnn import CNN
from src.hydra.hydra import Hydra
from src.data_factory.percolation import generate_percolation_data

def main(args):
    
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    real_images, _ = generate_percolation_data(
        dataset_size=args.dataset_size,
        lattice_size=args.lattice_size,
        p_list=[args.wanted_p],
        split=False,
        save_dir=None
    )

    cnn_checkpoint = torch.load(args.CNN_model_path, map_location=torch.device(args.device))
    cnn_checkpoint['constructor_args']['device'] = args.device
    cnn = CNN(**cnn_checkpoint['constructor_args'])
    cnn.load_state_dict(cnn_checkpoint['model_state_dict'])

    model = Hydra(
        cnn=cnn,
        lattice_size=args.lattice_size,
        noise_dim=args.noise_dim,
        n_conv_cells=args.n_conv_cells,
        n_convt_cells=args.n_convt_cells,
        generator_learning_rate=args.generator_learning_rate,
        discriminator_learning_rate=args.discriminator_learning_rate,
        l1=args.regularization_strength,
        l2=args.hydra_ratio_bce,
        l3=args.hydra_ratio_cnn,
        patience_generator=args.patience_generator,
        device=args.device,
        wanted_p=args.wanted_p,
        save_dir=save_dir
    )

    model._train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        real_images=real_images,
        set_generate_plots=args.set_generate_plots,
        bins_number=args.bins_number
    )

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--dataset_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_conv_cells", type=int, default=3)
    parser.add_argument("--n_convt_cells", type=int, default=5)
    parser.add_argument("--bins_number", type=int, default=100)
    parser.add_argument("--generator_learning_rate", type=float, default=10e-3)
    parser.add_argument("--discriminator_learning_rate", type=float, default=10e-3)
    parser.add_argument("--regularization_strength", type=float, default=1.0)
    parser.add_argument("--hydra_ratio_bce", type=float, default=1.0)
    parser.add_argument("--hydra_ratio_cnn", type=float, default=1.0)
    parser.add_argument("--patience_generator", type=int, default=2)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--wanted_p", type=float, default=0.5928)
    parser.add_argument("--save_dir", type=str, default="./saved_models/hydra")
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/cnn/2022.02.11.18.36.08/model/final_model.pt")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument('--set_generate_plots', dest='set_generate_plots', action='store_true')
    parser.add_argument('--no-set_generate_plots', dest='set_generate_plots', action='store_false')
    parser.set_defaults(set_generate_plots=False)
    args = parser.parse_args()
    main(args)
