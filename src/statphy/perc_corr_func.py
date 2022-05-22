# correlation function, connectivity function

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import numpy.random as random

from numba import jit
from math import log10, ceil, hypot



@jit
def perc_corr_func(m, lw):
    lx, ly = lw.shape
    L = max(lx,ly)
    r = np.arange(2 * L) # Positions
    pr = np.zeros(2 * L) # Correlation function
    npr = np.zeros(2 * L) # Nr of elements
    for ix1 in range(nx):
        for iy1 in range(ny):
            lw1 = lw[ix1,iy1]
            if (lw1>0):
                for ix2 in range(nx):
                    for iy2 in range(ny):
                        lw2 = lw[ix2,iy2]
                        if (lw2>0):
                            dx = (ix2-ix1)
                            dy = (iy2-iy1)
                            rr = hypot(dx, dy)
                            nr = int(ceil(rr)+1) # Corresponding box
                            pr[nr] = pr[nr] + (lw1==lw2)
                            npr[nr] = npr[nr] + 1
    pr = pr / npr
    return r, pr

#######################################

if __name__ == '__main__':
    from fit import loglog_slope

    # Calculate correlation function
    M = 10 # Nr of samples
    L = 128 # System size
    pp = [0.52,0.54,0.55,0.56,] # p-value 0.52,0.54,0.55,0.56
    lenpp = len(pp)
    pr = np.zeros((2*L,lenpp),float)
    rr = np.zeros((2*L,lenpp),float)
    for i in range(M):
        print("i = ",i)
        z = random.rand(L, L)
        for ip in range(lenpp):
            p = pp[ip]
            m = z < p
            lw, num = measurements.label(m)
            r, g = perccorrfunc(m, lw)
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