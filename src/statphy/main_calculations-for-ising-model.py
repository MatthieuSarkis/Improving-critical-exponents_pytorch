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
fake_imgs_file = f'./generated_data/fake-denoising-diffusion-pytorch/ising/2023.03.06.23.10.08//L={L}_p={p}.npy'
real_imgs_file = f'./generated_data/real/ising//L={L}_p={p}.bin'
max_n_samples = 5000
OUPUT_DIR_figs = 'output_files-ising/fig'
OUPUT_DIR_data = 'output_files-ising/txt'
read_real_images = False

clustering_sample_images = True
calc_stat_of_real_imgs = False
calc_stat_of_fake_imgs = False

######################## IMPORT MODULES #########################
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import geometric_measure
import plot_func
import sys
from modules import gen_class


######################## USEFULL FUNCTIONS ######################

def do_all_statistics_jobs(img_gen, img_shape, suffix='real'):

    L = img_shape[0]
    n_samples = img_gen.len()
    # get the mesures related to the configurations
    measure = geometric_measure.get_measure(img_gen, img_shape=img_shape)
    # get the statistics of measures
    stat = geometric_measure.measure_statistics(measure, nbins_for_ns=43)

    # part ns

    datainfo = f'p={p},L={L},N={n_samples}'

    ns = stat['ns']
    x, y, dx = ns['bin_centers'], ns['hist'], ns['bin_sizes']
    with open(f'{OUPUT_DIR_data}/ns_{suffix}({datainfo}).dat', 'w') as file:
        for z in zip(x, y, dx):
            file.write(f'{z[0]}\t{z[1]}\t{z[2]}\n')
    plt.figure()
    plot_func.logplotXY(plt, x, y, 
                    sim_st=f'$L={L}$', xlabel='$s$', ylabel='$n(s)$', 
                    xlow = 1e1, xup = 1e2, slope_st = '\\tau', show_legend=True, 
                    outfilename = f'{OUPUT_DIR_figs}/ns_{suffix}({datainfo}).pdf',)
    plt.close()
    
    # part gr
    x, y = stat['gr'] # r, gr
    with open(f'{OUPUT_DIR_data}/gr_{suffix}({datainfo}).dat', 'w') as file:
        for z in zip(x, y):
            file.write(f'{z[0]}\t{z[1]}\n')
    plt.figure()
    plot_func.logplotXY(plt, x, y, 
                    sim_st=f'$L={L}$', xlabel='$r$', ylabel='$g(r)$',
                    scale_xy_logplot= 1.05,
                    show_slope=True, xlow=1, xup=4, slope_st='\\eta' ,
                    precision=3,
                    marker='.', markersize=None, show_legend=True, 
                    outfilename=f'{OUPUT_DIR_figs}/gr_{suffix}({datainfo}).pdf')
    plt.close()

######################## MAIN FUNCTION ######################
if __name__ == '__main__':

    print(50*'-')

    fakedata, realdata = None, None

    if not os.path.exists(fake_imgs_file):
        print(f'File {fake_imgs_file} does not exist!')
    else:
        fakedata = np.load(fake_imgs_file)
        print(f'fakedata.shape={fakedata.shape}')
        print(fakedata[0])
        print(fakedata[0].min(), fakedata[0].max())

    if not os.path.exists(real_imgs_file):
        print(f'File {real_imgs_file} does not exist!')
    else:
        realdata = np.fromfile(real_imgs_file, dtype=np.int8) # shape=n*L*L
        realdata = realdata.reshape((-1, L, L))
        print(f'realdata.shape={realdata.shape}')
        print(realdata[0])
        sys.exit()

   
    os.makedirs(OUPUT_DIR_figs, exist_ok=True)
    os.makedirs(OUPUT_DIR_data, exist_ok=True)
    
    print(50*'-')
 


    #### CLUSTERING THE TEST IMAGES ###
    if clustering_sample_images:
        print ('Clustering some of real/fake images ...')
        n_samples = 5

        if fakedata is not None:
            imgs = fakedata[:n_samples]
            plt.figure()
            labels, _ = geometric_measure.clustering(imgs, lower_size=5)
            plot_func.plot_imgs_labels(plt, imgs, labels, outfilename=f'{OUPUT_DIR_figs}/imgs_fake(L={L}).pdf')
            plt.close()

        if realdata is not None:
            imgs = realdata[:n_samples]
            plt.figure()
            labels, _ = geometric_measure.clustering(imgs, lower_size=5)
            plot_func.plot_imgs_labels(plt, imgs, labels, outfilename=f'{OUPUT_DIR_figs}/imgs_real-L={L}_p={p}.pdf')
            plt.close()



    #### DO STATISTICS ###
    if calc_stat_of_real_imgs:
        print ('Doing calculations on the real images ...')

        if read_real_images_from_dir and filelist_real and len(filelist_real) > 0:
            img_gen = gen_class.GenUsingFile(filelist_real, max_n_samples)
        else:
            np.random.seed(72)
            img_gen = gen_class.GenUsingFunc(make_perc_func(L, p), max_n_samples)
        do_all_statistics_jobs(img_gen, img_shape=(L,L), suffix='real')

    if calc_stat_of_fake_imgs:
        if len(filelist_fake) > 0:
            print ('Doing calculations on the fake images ...')
            img_gen = gen_class.GenUsingFile(filelist_fake, max_n_samples)
            do_all_statistics_jobs(img_gen, img_shape=(L,L), suffix='fake')



