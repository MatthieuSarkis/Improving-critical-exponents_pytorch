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
import torch
import numpy as np
import os 

from src.hydra.hydra import Hydra

def main(args):

    os.makedirs(args.data_dir, exist_ok=True)
    
    #hydra_model_path = "./saved_models/hydra/2021.10.22.12.24.41/model/final_model.pt"
    #hydra_checkpoint = torch.load(hydra_model_path, map_location=torch.device(device))
    #hydra_checkpoint['constructor_args']['device'] = 'cpu'
    #hydra = Hydra(**hydra_checkpoint['constructor_args'])   
    #hydra.generator.load_state_dict(hydra_checkpoint['generator_state_dict'])
    #hydra.generator.eval()

    hydra = torch.load(args.model_dir)
    hydra.generator.eval()

    with torch.no_grad:
        for i in range(args.number_images):
            noise = torch.randn(1, args.noise_dim)
            image = hydra.generator(noise)
            image = torch.sign(image).to(torch.int8)
            image = image.numpy()
            np.save('./data/generated/{}'.format(i), image)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--number_images", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="./data/generated")
    parser.add_argument("--model_dir", type=str, default="./saved_models/hydra/2021.10.17.18.32.10/model/final_model.pt")
    parser.add_argument("--noise_dim", type=int, default=100)

    args = parser.parse_args()
    main(args)
