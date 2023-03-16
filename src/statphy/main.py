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


###################### INITIAL PARAMETERS, JUST CHANGE THIS PART #####################
p = 2.2692
L = 32
idir_fake = f'./generated_data/fake-denoising-diffusion-pytorch/ising/2023.03.06.23.10.08'
idir_real = f'./generated_data/real/ising'
max_n_samples = 5000
odir_figs = 'output_files-ising/fig'
odir_data = 'output_files-ising/txt'

clustering_sample_images = True
calc_statistics = True

######################## IMPORT MODULES #########################
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import geometric_measure
import plot_func
from modules import gen_class


######################## USEFULL FUNCTIONS ######################

def get_data(idir, sep='_', ext='npy'):
    """
    read all files matched with ext and append the read data, and return final data
    """
    filepattern = f'*L{sep}{L}_p{sep}{p}*.{ext}'
    filelist = glob.glob(os.path.join(idir, filepattern))

    data = np.empty((0,L,L), np.int8)

    for path in filelist:
        if ext == 'npy':
            curr = np.load(path)
        elif ext == 'bin':
            curr = np.fromfile(path, np.int8).reshape((-1,L,L))
        else:
            pass
        data = np.append(data, curr, axis=0)
    
    return data



def do_all_jobs(img_gen, suffix='real'):

    n_samples = len(img_gen)


    # get the statistics of measures
    measure_obj = geometric_measure.Measure()

    history = measure_obj.calc_history(img_gen, (L,L), 1, True)
    stats = measure_obj.get_stats(nbins_for_ns=33, nbins_for_m_gr=33)


    filenamesuffix = f'(L={L},p={p},n={n_samples})--{suffix}'
    plot_func.plot_stats(plt, stats, odir_figs, filenamesuffix, f'$L={L}$' )


######################## MAIN FUNCTION ######################
if __name__ == '__main__':

   

    fakedata, realdata = None, None

    if not os.path.exists(idir_fake):
        print(f'dir {idir_fake} does not exist!')
    else:
        fakedata = get_data(idir_fake, ext='npy')
        print(f'fakedata.shape={fakedata.shape}')

    if not os.path.exists(idir_real):
        print(f'dir {idir_real} does not exist!')
    else:
        realdata = get_data(idir_real, ext='bin')
        realdata = realdata.reshape((-1, L, L))
        print(f'realdata.shape={realdata.shape}')

   
    os.makedirs(odir_figs, exist_ok=True)
    os.makedirs(odir_data, exist_ok=True)
    
    print(50*'-')
 


    #### CLUSTERING SOME OF IMAGES ###
    if clustering_sample_images:
        print ('Clustering some of real/fake images ...')
        n_samples = 5

        if fakedata is not None:
            imgs = fakedata[:n_samples]
            plt.figure()
            labels, _ = geometric_measure.clustering(imgs, lower_size=5)
            plot_func.plot_imgs_labels(plt, imgs, labels, outfilename=f'{odir_figs}/imgs_fake(L={L}).pdf')
            plt.close()

        if realdata is not None:
            imgs = realdata[:n_samples]
            plt.figure()
            labels, _ = geometric_measure.clustering(imgs, lower_size=5)
            plot_func.plot_imgs_labels(plt, imgs, labels, outfilename=f'{odir_figs}/imgs_real-L={L}_p={p}.pdf')
            plt.close()



    #### DO STATISTICS ###
    if calc_statistics:
        
        if realdata is not None:
            print ('Calculating the statsitics of real images ...')

            img_gen = gen_class.GenUsingList(realdata, max_n_samples)
            do_all_jobs(img_gen, suffix='real')
            

    if calc_statistics and fakedata is not None:
        pass


        # if fakedata is not None:
            # print ('Calculating the statsitics of fake images ...')

            # img_gen = gen_class.GenUsingList(realdata, max_n_samples)
            # do_all_jobs(img_gen, suffix='fake')

