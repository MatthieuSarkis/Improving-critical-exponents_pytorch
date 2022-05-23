# -*- coding: utf-8 -*-
#
# Written by Hor (Ebi) Dashti, https://github.com/h-dashti
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# correlation function, connectivity function

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

from numba import jit
import math

@jit
def coordinate(i : int, ly : int) -> tuple[int, int]:
    x = i // ly
    y = i - x * ly
    return x, y

@jit
def corr_func(label : np.ndarray, max_r : int, n_trials : int) -> np.ndarray:
    lx, ly = label.shape
    cr = np.zeros(max_r) # count of events
    Nr = np.zeros(max_r) # count of samples

    for k in range(n_trials):
        i1 = np.random.randint(lx * ly)
        i2 = np.random.randint(lx * ly)
        if i1 == i2: 
            continue
        x1, y1 = coordinate(i1, ly)
        x2, y2 = coordinate(i2, ly)
        c1 = label[x1, y1]
        c2 = label[x2, y2]
        if c1 >= 0 and c2 >= 0:
            dx, dy = x2 - x1, y2 - y1
            dr = math.hypot(dx, dy)
            ir = round(dr) # Corresponding box
            cr[ir] = cr[ir] + (c1 == c2)
            Nr[ir] = Nr[ir] + 1

    return np.where(Nr == 0, 0, cr / Nr)  # cr / Nr

@jit
def corr_func_exact_method(label : np.ndarray, max_r : int):
    lx, ly = label.shape
    cr = np.zeros(max_r) # count of events
    Nr = np.zeros(max_r) # count of samples

    for x1 in range(lx):
        for y1 in range(ly):
            c1 = label[x1, y1]
            if c1 >= 0:
                for x2 in range(lx):
                    for y2 in range(ly):
                        c2 = label[x2, y2]
                        if c2 >= 0:
                            dx, dy = x2 - x1, y2 - y1
                            dr = math.hypot(dx, dy)
                            ir = round(dr) #math.ceil(dr) + 1 # Corresponding box
                            cr[ir] = cr[ir] + (c1 == c2)
                            Nr[ir] = Nr[ir] + 1

    return np.where(Nr == 0, 0, cr / Nr)  # cr / Nr


if __name__ == '__main__':
    import sys
    import plots

    # Calculate correlation function
    n_samples = 1000 # number of samples
    L = 128 
    p_arr = [0.59,] # p-value 0.52,0.54,0.55,0.56
    len_p_arr = len(p_arr)
    max_r = 2 * L
    gr_accum = np.zeros((max_r, len_p_arr), dtype=float)
    r = np.arange(max_r, dtype=int)
    np.random.seed(72)

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"sample {i+1}/{n_samples} ", flush=True)

        for ip in range(len_p_arr):
            p = p_arr[ip]
            mat = (np.random.random(size=(L,L)) < p).astype(int)
            labeled, num = measurements.label(mat)
            labeled = labeled - 1 # shift value of labels to make the computation easier
            gr = corr_func(labeled, max_r=2*L, n_trials=200*L**2)
            gr_accum[:,ip] = gr_accum[:,ip] + gr

    gr_accum = gr_accum / n_samples

    # Plot data - linearly binned
    for ip in range(len_p_arr):
        x, y = r, gr_accum[:,ip]
        indx = np.nonzero(y)
        x, y = x[indx], y[indx]

        plots.logplotXY(plt, x, y, sim_st=f"$p={p_arr[ip]}$", 
                        scale_xy_logplot= 1.05,
                        show_slope=True, xlow=3, xup=10, slope_st='\\eta' ,
                        marker='.', markersize=None)

    plt.xlabel('$r$', fontsize=16)
    plt.ylabel('$g(r)$', fontsize=16)
    plt.legend()
    plt.savefig(f'gr(L={L},N={n_samples}).pdf', pad_inches=0.01, bbox_inches='tight' )