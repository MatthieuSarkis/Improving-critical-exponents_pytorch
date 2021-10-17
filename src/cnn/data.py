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

import numpy as np
import torch
from torch.utils.data import Dataset

from src.statphy.models.percolation import percolation_configuration

def generate_data(dataset_size, lattice_size=128):

    X = []
    y = []

    for _ in range(dataset_size):
        y.append(np.random.rand())
        X.append(percolation_configuration(lattice_size, y[-1]))

    X = np.array(X)
    y = np.array(y)

    return X, y

class LatticeConfigurations(Dataset):

    def __init__(self, 
                 dataset_size: int,
                 lattice_size: int = 128, 
                 transform=None,
                 ) -> None:
        
        self.dataset_size = dataset_size
        self.lattice_size = lattice_size
        
        X, y = generate_data(dataset_size=dataset_size, lattice_size=lattice_size)
        self.X = torch.tensor(X).float().unsqueeze(1)
        self.y = torch.tensor(y).float().view(-1, 1)
        
        self.transform = transform

    def __len__(self):
        
        return self.dataset_size

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'images': self.X[idx], 
                  'labels': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)
            
        return sample

def generate_data_torch(dataset_size, lattice_size=128):

    X = []
    y = []

    for _ in range(dataset_size):
        y.append(np.random.rand())
        X.append(percolation_configuration(lattice_size, y[-1]))

    X = torch.tensor(X).float().unsqueeze(1)
    y = torch.tensor(y).float().view(-1, 1)
    
    X_train = X[:(3*dataset_size)//4]
    X_test = X[(3*dataset_size)//4:]
    y_train = y[:(3*dataset_size)//4]
    y_test = y[(3*dataset_size)//4:]

    return X_train, y_train, X_test, y_test