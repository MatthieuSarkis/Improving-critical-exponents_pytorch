# -*- coding: utf-8 -*-
#
# Written by Hor (Ebi) Dashti, https://github.com/h-dashti
#


import scipy.stats as stats
import numpy as np


def loglog_slope(x, y, full=True):

    x, y = np.log(x), np.log(y)

    if full:
        popt, pcov = np.polyfit(x, y, deg=1, cov=True)
        perr = np.sqrt(np.diag(pcov))
    else:
        popt = np.polyfit(x, y, deg=1, cov=False)
       
    expo = popt[0]
    c = np.exp(popt[1])
    
    if full:
        return expo, c, perr[0], c * perr[1]
    else:
        return expo, c
        
        
def loglog_cov(x, y):

    x, y = np.log(x), np.log(y)

    popt, pcov = np.polyfit(x, y, deg=1, cov=True)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def lin_slope(x, y, full=True):

    if full:
        popt, pcov = np.polyfit(x, y, deg=1, cov=True)
        perr = np.sqrt(np.diag(pcov))
    else:
        popt = np.polyfit(x, y, deg=1, cov=False)
    
    if full:
        return popt[0], popt[1], perr[0], perr[1] 
    else:
        return popt


def linregress(x, y):
    return stats.linregress(x, y)




if  __name__ == '__main__':
    pass