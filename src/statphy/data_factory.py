# -*- coding: utf-8 -*-
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generate data for specify model."""

from argparse import ArgumentParser
import os 
import numpy as np
import itertools

from src.statphy.models.percolation import percolation_configuration

_available_models = [
    "square_lattice_percolation",
    ]

def main(args):

    os.makedirs(args.path, exist_ok=True)

    if args.model == "square_lattice_percolation":

        for L, p in itertools.product(args.L, args.control_parameter):

            print ('\nGenerating data for lattice size L={} and control parameter p={}\n'.format(L, p))

            X = np.array([percolation_configuration(L, p) for __ in range(args.samples)],dtype='float32').reshape(args.samples,L,L,1)
            path = args.path + "/L={}_p={}.npz".format(L, p)
            np.savez(path, X)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model", type=str, default="square_lattice_percolation", 
                        choices=_available_models)

    # Model Parameters
    parser.add_argument("--L", type=int, nargs='+', default=[64, 128])
    parser.add_argument("--control_parameter", type=float, nargs='+', default=[0.5, 0.6])

    # Statistics
    parser.add_argument("--samples", type=int, default=10)

    # Save data
    parser.add_argument("--path", type=str, default="./data/simulation")

    args = parser.parse_args()
    main(args)