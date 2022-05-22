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
from math import ceil, hypot

def coordinate(i, ly):
    x = i / ly
    y = i - x * ly
    return x, y

@jit
def perc_corr_func(label):
    lx, ly = label.shape
    n_trails = 100 * lx * ly
    L = max(lx,ly)
    r = np.arange(2 * L) # Positions
    pr = np.zeros(2 * L) # Correlation function
    npr = np.zeros(2 * L) # Nr of elements

    for k in range(n_trails):
        i1 = np.random.randint(lx * ly)
        i2 = np.random.randint(lx * ly)
        x1, y1 = coordinate(i1, ly)
        x2, y2 = coordinate(i2, ly)
        c1, c2 = label[x1, y1], label[x2, y2]
        if c1 >= 0 and c2 >= 0:
            dx, dy = x2 - x1, y2 - y1
            rr = hypot(dx, dy)
            nr = int(ceil(rr) + 1) # Corresponding box
            pr[nr] = pr[nr] + (c1 == c2)
            npr[nr] = npr[nr] + 1
    
    pr = pr / npr
    return r, pr


if __name__ == '__main__':
    from fit import loglog_slope

    # Calculate correlation function
    M = 10 # number of samples
    L = 32 
    pp = [0.59,] # p-value 0.52,0.54,0.55,0.56
    lenpp = len(pp)
    pr = np.zeros((2*L,lenpp),float)
    rr = np.zeros((2*L,lenpp),float)
    for i in range(M):
        print("i = ",i)
        z = np.random.rand(L, L)
        for ip in range(lenpp):
            p = pp[ip]
            m = z < p
            lw, num = measurements.label(m)
            r, g = perc_corr_func(m, lw)
            pr[:,ip] = pr[:,ip] + g
            rr[:,ip] = rr[:,ip] + r
    pr = pr / M
    rr = rr / M

    # Plot data - linearly binned
    for ip in range(lenpp):
        x, y = rr[:,ip], pr[:,ip]

        plt.loglog( x, y, '.', label = f"$p={pp[ip]}$")

        if False:
            indx = (x > 3) & (x < 50)
            xn, yn = x[indx], y[indx]

            slope, slope_err, c, c_err = loglog_slope(xn, yn)
            print (f'slope={slope:.2f}, err={slope_err:.2f}')

            #xn = np.logspace(log10(xn.min()), log10(xn.max()))
            #yn = c * xn**slope
            #plt.loglog(xn, yn)


    plt.legend()
    plt.savefig(f'Gr_{L}.pdf',
                pad_inches=0.02, bbox_inches='tight' )