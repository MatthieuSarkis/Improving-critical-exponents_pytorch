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

""" Computing the geometric measures of clusters """

import numpy as np
from scipy.ndimage import measurements
import itertools
import gen_class
from numba import jit
import math

def clustering(imgs, 
               target_color = 1, max_num = -1, 
               change_type = 'shuffle', 
               lower_size = -1):
    """
    Color the list of images and return a list of colored 2d arrays
    """
    labels, nums = [], []
    i = 0
    for img in imgs:
        site = np.where(img == target_color, 1, 0)
        label, numbers = measurements.label(site)

        if change_type == 'shuffle':
            index = np.arange(1, label.max() + 1) 
            np.random.shuffle(index) 
            index = np.insert(index, 0, 0)
            label = index[label]
            area = measurements.sum(site, label, index=np.arange(label.max() + 1) ).astype(int)
            label = np.where(area[label] < lower_size, 0, label)

        labels.append(label)
        nums.append(numbers)
        i += 1
        if max_num == i:
            break

    return labels, nums


def get_measure(img_gen : gen_class, 
                img_shape,
                target_color = 1,
                show_progress = True,
               ):
    """
    Compute the geometric measures of a serie of configurations
    and put in a list.

    Parameters
    ----------
    img_gen : generator of images
        images are with integer values.
    target_color : int, default=1
        the target value to be considered for clustering
    show_progress : bool, default=true
        If ture, then a progress bar is shown.

    Returns
    ----------
    measure : dictionary which contains
        all_mass : array int
            list of mass of each cluster over all configurations
        all_Rs2 : array float
            list of gyration radius of each cluster over all configurations
        all_chi : array float
            list of chi over all configurations.
            chi is the average size of a cluster connected to random point. 
        all_xi : array float
            list of xi over all configurations.
            xi is the correlation length
        all_big : array int
            list of biggest cluster over all configurations.
            big is the size of biggest cluster for each configruation.
        all_M : array int
            list of M over all configurations.
            M is the mass of the spanning (percolated) cluster for each configruation.
            If we dont have an spanning cluster, then M=0
   
    """
    N = img_gen.len()      # the number of images

    if show_progress:
        import tqdm
        myrange = tqdm.trange(N)
    else:
        myrange = range(N)

    all_mass = np.array([], dtype=int) 
    all_Rs2 = np.array([], dtype=float)
    all_chi = np.zeros(N, dtype=float)
    all_big = np.zeros(N, dtype=int)
    all_M = np.zeros(N, dtype=int)
    all_xi = np.zeros(N, dtype=float)
    
    for ii in myrange:
        img = next(img_gen)
        # generate an array of target color
        site = np.where(img == target_color, 1, 0)

        # finding the label of each cluster
        label, _ = measurements.label(site)
        label = label - 1 # note that for better calculation, we shift labels' values

        # a range array start from 1 to max(label); note that index zero is not our concern
        mlabel_list = np.arange(label.max()+1)

        # mass(size) of each cluster;
        mass = measurements.sum(site, label, index=mlabel_list).astype(int)
        all_mass = np.append(all_mass, mass)   

        # biggest cluster size
        all_big[ii] = np.max(mass)
    
        # center of mass of each cluster
        cm = measurements.center_of_mass(site, label, index=mlabel_list)
       
        # calculate the gyration radius for each cluster
        rs2 = np.zeros(len(mlabel_list), dtype=float) 
        for i, j in itertools.product(range(img_shape[0]), range(img_shape[1])):
            indx = label[i, j] # the label at position (i, j)
            if indx >= 0:
                dr = np.array([i, j]) - cm[indx]
                dr2 = np.dot(dr, dr)
                rs2[indx] = rs2[indx] + dr2
        rs2 = rs2 / mass 
        all_Rs2 = np.append(all_Rs2, rs2)

        # find the spanning cluster along x axis
        # since there is no preference for axes, we just consider x axis
        perc_x = np.intersect1d(label[0,:], label[-1,:])
        perc = perc_x[np.where(perc_x >= 0)] 
       
        if len(perc) > 0: 
            all_M[ii] = mass[perc[0]]
            mass[perc[0]] = 0 # remove spanning cluster by setting its mass to zero
    
        msum  = np.sum(mass)
        msum2 = np.sum(mass * mass)
        if msum > 0:
            all_chi[ii] = msum2 / msum                          #  chi = [sum s^2] / [sum s]
            all_xi[ii] = np.sum(2 * rs2 * mass * mass) / msum2  #  xi = [sum 2*Rs^2 * s^2] / [sum s^2]

    measure = {
        'N' : N,
        'shape' : img_shape,
        'all_mass' : all_mass,
        'all_Rs2' : all_Rs2,
        'all_chi' : all_chi, 
        'all_xi' : all_xi, 
        'all_big' : all_big, 
        'all_M' : all_M,
        }

    return measure


def cluster_number_density(all_mass, img_size, N, 
                           nbins=None, **kwargs):
    """
    Calculation of cluster number density n(s);
    n(s) = N(s) / (img_size * N), 
    where N(s) is histogram of cluster size s, 
    and N is the number of configurations, 
    and img_size is the size of image (lattice)

    Parameters
    ----------
    all_mass : array int
        list of area of clusters over all configurations.
    img_size : int
        the size of each image which is Lx*Ly.
    N : int
        number of images
    nbins : int, default=None
        number of bins for calculating the histogram of n(s).
        
    Returns
    ----------
    If nbins is None, return n(s), 
    otherwise return the histogram of n(s).
    """
    # the number of clusters of size s measured in N realizations 
    Ns = np.bincount(all_mass) 

    # the cluster number density
    # Note that ns at some indexes may has the zero value
    ns = Ns.astype(float) / (img_size * N)

    if nbins is None:
        return ns
    else:
        import histogram

        s = np.arange(len(ns))
        # we dont need the ns at index zero
        return histogram.hist(s[1:], ns[1:], nbins=nbins, **kwargs) 
    

def measure_statistics(measure,
                       nbins_for_ns = 53):
    """
    This calculates the statistics of measures

    Parameters
    ----------
    measure : dictionay
        measure for clusters.
    nbins_for_ns : int
        number of bins for calculation of cluster number density

    Returns
    ----------
    stat : dictionary
    """

    N = measure['N']
    img_size = np.product(measure['shape'])

    stat = {}
    ns = cluster_number_density(measure['all_mass'], 
                                img_size = img_size, 
                                N = N, 
                                nbins = nbins_for_ns
                               )
    stat['ns'] = ns
    stat['chi'] = np.average(measure['all_chi'])
    stat['xi'] = np.average(measure['all_xi'])
    stat['Pinf'] = np.average(measure['all_big']) / img_size
    stat['M'] = np.average(measure['all_M'])
    stat['Ps'] = np.average(measure['all_M'] > 0)

    return stat


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
            rr = math.hypot(dx, dy)
            nr = int(math.ceil(rr) + 1) # Corresponding box
            pr[nr] = pr[nr] + (c1 == c2)
            npr[nr] = npr[nr] + 1
    
    pr = pr / npr
    return r, pr


if __name__ == '__main__':

    img1 = np.array([[0,0,1,1,0,1],
                     [0,0,1,0,1,1],
                     [1,1,1,0,1,0],
                     [0,0,1,1,0,1]])
    img2 = np.array([[0,1,1,0,0,0],
                     [0,1,1,1,0,0],
                     [1,1,0,0,1,0],
                     [0,1,0,1,0,0]])

    import gen_class
    img_gen = gen_class.GenUsingList([img1, img2], 2)

    # get the mesures related to the configurations
    measure = get_measure(img_gen, img_shape=img1.shape)
    # get the statistics of measures
    stat = measure_statistics(measure)