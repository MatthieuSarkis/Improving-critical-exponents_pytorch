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

import os
import numpy as np
import torch

def percolation_configuration(L, p):
    spin = (np.random.random(size=(L,L)) < p).astype(np.int8)
    return 2 * spin - 1

def generate_data_torch(dataset_size, lattice_size=128, split=False, save_dir=None):

    X = []
    y = []

    for _ in range(dataset_size):
        y.append(np.random.rand())
        X.append(percolation_configuration(lattice_size, y[-1]))

    X = np.array(X)
    y = np.array(y)

    X = torch.from_numpy(X).float().unsqueeze(1)
    y = torch.from_numpy(y).float().view(-1, 1)
    
    if save_dir is not None:
        torch.save(X, os.path.join(save_dir, 'images.pt'))
        torch.save(y, os.path.join(save_dir, 'labels.pt'))
    
    if split:
        X_train = X[:(3*dataset_size)//4]
        X_test = X[(3*dataset_size)//4:]
        y_train = y[:(3*dataset_size)//4]
        y_test = y[(3*dataset_size)//4:]
 
    return (X, y) if not split else (X_train, y_train, X_test, y_test)