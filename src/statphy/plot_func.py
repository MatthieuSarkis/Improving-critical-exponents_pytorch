import numpy as np
import math
from modules import fit


def logspace_indxs(x, scale):
    x0, x1 = 1, len(x)
    space = np.logspace(math.log(x0, scale), math.log(x1, scale),
                        base=scale, dtype=int)
    return np.unique(space) - 1

def logspace_XY(x, y, scale):
    indxs = logspace_indxs(x, scale)
    return x[indxs], y[indxs]

def plot_imgs_labels(plt, imgs, labels, outfilename):
    nrows = len(imgs)
    plt.figure(figsize=(2*3, nrows*3))
    plt.subplots_adjust(wspace=0, hspace=0.01)
    
    for i, img in enumerate(imgs):
        plt.subplot(nrows, 2, 2*i + 1)
        plt.imshow(img, cmap='Greys',)
        plt.axis('off')
        plt.subplot(nrows, 2, 2*i + 2)
        label = labels[i]
        plt.imshow(label,)
        plt.axis('off')
    if outfilename:
        plt.savefig(outfilename, pad_inches=0.01, bbox_inches='tight') #

def logplotXY(plt, x, y, xlabel=None, ylabel=None, title=None, outfilename=None,
              show_legend=False,
              sim_st = 'sim',
              scale_xy_logplot = 1,
              show_slope = True,
              xlow = 1e1, xup = 1e3,
              slope_st = '\\tau',
              marker = 'o' ,
              markersize=5,
              precision = 2
            ):

    if scale_xy_logplot <= 1:
        xn, yn = x, y
    else:
        xn, yn = logspace_XY(x, y, scale_xy_logplot)   
           
    pl = plt.loglog(xn, yn, ls='', marker=marker, 
                    markersize = markersize,
                    label = sim_st)

    if show_slope:
        indx = (x >= xlow) & (x <= xup)
        xn, yn = x[indx], y[indx]
        expo, c, expo_err, c_err = fit.loglog_slope(xn, yn)
        xn = np.logspace(np.log10(xn.min()), np.log10(xn.max()))
        yn = c * xn ** expo
        expo_usign = expo if expo > 0 else -expo
        plt.loglog(xn, yn, color=pl[0].get_color(),
                label = fr'${slope_st}={expo_usign:.{precision}f} \pm {expo_err:.{precision}f}$'  )
    
    if show_legend:
        plt.legend(frameon=False)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel, fontsize=16)
    if ylabel:
        plt.ylabel(ylabel, fontsize=16)
    if outfilename:
        plt.savefig(outfilename, pad_inches=0.01, bbox_inches='tight')
