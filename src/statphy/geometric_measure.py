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
#from scipy.ndimage import measurements
import scipy.ndimage as imgprocessing
import itertools
from numba import jit

from modules import gen_class
from modules import histogram
import perc_corr
import sys


def clustering(imgs, 
               target_spin = 1, max_num = -1, 
               change_type = 'shuffle', 
               lower_size = -1):
    """
    Color the list of images and return a list of colored 2d arrays
    """
    labels, nums = [], []
    i = 0
    for img in imgs:
        mat = np.where(img == target_spin, 1, 0)
        label, numbers = imgprocessing.label(mat)

        if change_type == 'shuffle':
            index = np.arange(1, label.max() + 1) 
            np.random.shuffle(index) 
            index = np.insert(index, 0, 0)
            label = index[label]
            area = imgprocessing.sum(mat, label, index=np.arange(label.max() + 1) ).astype(np.int32)
            label = np.where(area[label] < lower_size, 0, label)

        labels.append(label)
        nums.append(numbers)
        i += 1
        if max_num == i:
            break

    return labels, nums


def bining_xy(X: np.ndarray, Y: np.ndarray, nbins: int = -1):
    '''
    It divides xrange to nbins and calculate the value of y_avg and x_avg for each bin.
    '''
    xmax, xmin = np.max(X), np.min(X)
    if nbins < 0:
        dx = 1
        nbins = (xmax - xmin)
    else:
        dx = (xmax - xmin) / nbins

    Xn, Yn, Cn = np.zeros(nbins+1), np.zeros(nbins+1), np.zeros(nbins+1, dtype=np.int64)

    for x, y in zip(X, Y):
        indx = int((x - xmin)/dx)
        Xn[indx] += x
        Yn[indx] += y
        Cn[indx] += 1
    
    nonzero_indxs = (Cn > 0)
    Cn = Cn[nonzero_indxs]
    Xn = Xn[nonzero_indxs] / Cn
    Yn = Yn[nonzero_indxs] / Cn

    return (Xn, Yn)
    




class Measure:
    def __init__(self, ) -> None:
        self.history = dict()
        self.stats = dict()
    
    def calc_history(self, 
                      img_gen : gen_class, 
                      img_shape: tuple[int, int], 
                      target_spin: int = 1, 
                      show_progress: bool = True
                    ):
        """
        Compute the geometric measures of a serie of configurations
        and put in a list.

        Parameters
        ----------
        img_gen : generator of images
            images are with integer values.
        target_spin : int, default=1
            the target value to be considered for clustering
        show_progress : bool, default=true
            If ture, then a progress bar is shown.

        Returns
        ----------
        measures : dictionary which contains
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
        n_imgs = len(img_gen)

        self.n_imgs = n_imgs      # the number of images
        self.img_shape = img_shape

        if show_progress:
            import tqdm
            myrange = tqdm.trange(n_imgs)
        else:
            myrange = range(n_imgs)

        all_mass = np.array([], dtype=np.int32) 
        all_Rs2 = np.array([], dtype=np.float64)
        all_chi = np.zeros(n_imgs, dtype=np.float64)
        all_big = np.zeros(n_imgs, dtype=np.int32)
        all_M = np.zeros(n_imgs, dtype=np.int32) # biggest cluster size
        all_xi = np.zeros(n_imgs, dtype=np.float64)

        # varibale for gr, the correlation function
        
        L = max(img_shape)
        max_r = 2 * L
        gr_accum = np.zeros(max_r, np.float64)
        
        count_samples = 0

        for ii in myrange:
            img = next(img_gen)

            # generate an array of target color
            mat = np.where(img == target_spin, 1, 0)

            # finding the label of each cluster
            label, _ = imgprocessing.label(mat)

            # we shift labels' values.
            # so the negative labels (that corresponds to non-intrested spins in mat) are not considered.
            label = label - 1 


            # an array [0, max(label)];
            mlabel_list = np.arange(label.max()+1)

            # mass/size of each cluster; clusters' index start from zero. negaitive lables are not considered.
            mass = imgprocessing.sum(mat, label, index=mlabel_list).astype(np.int32)

            
            all_mass = np.append(all_mass, mass)   

            # biggest cluster
            indx_big = np.argmax(mass)
            all_big[ii] = mass[indx_big]  # np.max(mass)
        
            # center of mass of each cluster
            cm = imgprocessing.center_of_mass(mat, label, index=mlabel_list)

        
            # calculate the gyration radius for each cluster
            rs2 = np.zeros(len(mlabel_list), dtype=np.float64) 
            for i, j in itertools.product(range(img_shape[0]), range(img_shape[1])):
                indx = label[i, j] # the label at position (i, j)
                if indx >= 0:
                    dr = np.array([i, j]) - cm[indx]
                    dr2 = np.dot(dr, dr)
                    rs2[indx] = rs2[indx] + dr2
            rs2 = rs2 / mass 
            all_Rs2 = np.append(all_Rs2, rs2)



            # find the spanning cluster along x axis
            perc_x = np.intersect1d(label[0,:], label[-1,:])
            perc = perc_x[np.where(perc_x >= 0)] 
            indx_perc = perc[0] if len(perc) > 0 else -1

            if indx_perc < 0: # if we could not find the spannig cluster along x, we search along y axis
                perc_y = np.intersect1d(label[:,0], label[:,-1])
                perc = perc_y[np.where(perc_y >= 0)] 
                indx_perc = perc[0] if len(perc) > 0 else -1
        
            if indx_perc > 0: 
                all_M[ii] = mass[indx_perc]
                mass[indx_perc] = 0 # remove spanning cluster by setting its mass to zero
        
            msum  = np.sum(mass)
            msum2 = np.sum(mass * mass)
            if msum > 0:
                all_chi[ii] = msum2 / msum                          #  chi = [sum s^2] / [sum s]
                all_xi[ii] = np.sum(2 * rs2 * mass * mass) / msum2  #  xi = [sum 2*Rs^2 * s^2] / [sum s^2]
            

            # calcualte gr
            gr = perc_corr.corr_func(label, max_r=max_r, n_trials=100*L**2)
            # if indx_perc < 0:
            #     gr = perc_corr.corr_func(label, max_r=max_r, n_trials=100*L**2)
            # else:
            #     #gr = perc_corr.corr_func(label, max_r=max_r, n_trials=100*L**2)
            #     gr = perc_corr.corr_func_ignore_a_clus(label, max_r=max_r, n_trials=200*L**2, indx_skip=indx_perc)
            gr_accum = gr_accum + gr


            count_samples += 1
        
        # end for loop
        
        # finalize the gr arrays

        self.history = {
            'n_samples' : count_samples,
            'shape' : img_shape,
            'all_mass' : all_mass,
            'all_Rs2' : all_Rs2,
            'all_chi' : all_chi, 
            'all_xi' : all_xi, 
            'all_big' : all_big, 
            'all_M' : all_M,
            'row_gr' : gr_accum / count_samples,
            }

        return self.history
    
    def get_stats(self,
            nbins_for_ns: int = 53,
            nbins_for_m_gr: int = 53, 
            ):
        """
        This calculates the statistics of measures

        Parameters
        ----------
        measure : dictionary
            measure for clusters.
        nbins_for_ns : int
            number of bins for calculation of cluster number density

        Returns
        ----------
        stats : dictionary
        """

        #n_imgs = self.n_imgs
        img_shape = self.img_shape
        img_size = np.product(img_shape)


        ns = self.__cluster_number_density(nbins = nbins_for_ns)
        
        self.stats['rg,m'] = bining_xy(np.sqrt(self.history['all_Rs2']), self.history['all_mass'], nbins=nbins_for_m_gr)

        self.stats['ns'] = ns 
        self.stats['chi'] = np.average(self.history['all_chi'])
        self.stats['xi'] = np.average(self.history['all_xi'])
        self.stats['Pinf'] = np.average(self.history['all_big']) / img_size

        # M is the size of percolated cluster; 
        # so if M=0, it means we do not have any percolated clusteer
        self.stats['M'] = np.average(self.history['all_M'])
        self.stats['Ps'] = np.average((self.history['all_M'] > 0))  

        gr = self.history['row_gr']
        r = np.arange(gr.size, dtype=np.int32)
        indx = np.nonzero(gr)
        self.stats['r,gr'] = (r[indx], gr[indx])

        return self.stats
    

    def __cluster_number_density(self,
            nbins: int=None, **kwargs
            ):
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
        n_imgs : int
            number of images
        nbins : int, default=None
            number of bins for calculating the histogram of n(s).
            
        Returns
        ----------
        If nbins is None, return n(s), 
        otherwise return the histogram of n(s).
        """
        # the number of clusters of size s measured in N realizations 
        Ns = np.bincount(self.history['all_mass']) 
    

        # the cluster number density
        # Note that ns at some indexes may has the zero value
        ns = Ns.astype(np.float64) / (np.product(self.img_shape) * self.n_imgs)

        if nbins is None or nbins <= 0:
            return ns
        else:
            s = np.arange(len(ns))
            # we dont need the ns at index zero
            return histogram.hist(s[1:], ns[1:], nbins=nbins, **kwargs) 




if __name__ == '__main__':

    img1 = np.array([[1,0,1,1,0,1],
                     [0,0,1,0,1,1],
                     [1,1,1,0,1,0],
                     [0,0,1,1,0,1]])
    img2 = np.array([[1,0,0,0,0,1],
                     [0,1,1,1,1,0],
                     [1,1,0,0,1,1],
                     [0,0,0,1,0,0]])

    from modules import gen_class
    img_gen = gen_class.GenUsingList([img1, img2], 2)



    # get the statistics of measures
    measure = Measure()

    measure.calc_history(img_gen, img1.shape, 1, False)
    
    stats = measure.get_stats()

    print(stats)

    