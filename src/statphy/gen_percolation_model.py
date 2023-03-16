# -*- coding: utf-8 -*-
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Square Lattice Percolation model."""

import numpy as np
import os
import itertools

def generate(L, p):
    spin = (np.random.random(size=(L,L)) < p).astype(np.int8)
    return spin

def generate_data(L, p_arr, max_configs_per_p=1000):
    X, y = [], []
    unique_labels = {}
  
    j = 0
    for p in p_arr:
        unique_labels[str(p)] = j
        for i in range(max_configs_per_p):
            X.append(generate(L, p))
            y.append(j)
        j += 1
    X = np.array(X).reshape(-1, L, L, 1)
    y = np.array(y).reshape(-1, )
    return X, y, unique_labels


if __name__ == '__main__':

    L_arr = [1024,]
    p_arr = [0.5927,]
    n_samples = 1000

    odir = 'generated_data/real/perc'
    os.makedirs(odir, exist_ok=True)

    for L, p in itertools.product(L_arr, p_arr):

        print(f'L={L}, p={p}')

        data = np.empty((n_samples,L,L), np.int8)

        for i in range(n_samples):
            data[i] = generate(L, p)

        np.save(f'{odir}/L_{L}_p_{p}.npy', data)