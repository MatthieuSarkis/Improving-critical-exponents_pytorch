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
import os 

from src.conv_vae.vae import Conv_VAE

def main(args):

    os.makedirs(args.data_dir, exist_ok=True)
    
    #device = 'cpu'
    #vae_checkpoint = torch.load(args.model_dir, map_location=torch.device(device))
    
    vae_checkpoint = torch.load(args.model_dir)
    vae_checkpoint['constructor_args']['device'] = 'cpu'
    vae = Conv_VAE(**vae_checkpoint['constructor_args'])   
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()

    vae.decoder.sample_images(
        n_images_per_p=args.n_images_per_p, 
        properties=args.properties, 
        directory_path=args.data_dir,
    )

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--n_images_per_p", type=int, default=8)
    parser.add_argument("--properties", nargs="*", type=float, default=0.5928)
    parser.add_argument("--data_dir", type=str, default="./data/conv_vae_generated")
    parser.add_argument("--model_dir", type=str, default="./saved_models/conv_vae/2022.02.17.13.26.37/Convolutional_VAE_model/final_model.pt")

    args = parser.parse_args()
    main(args)
