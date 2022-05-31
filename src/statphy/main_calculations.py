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


## INITIAL PARAMETERS
p = 0.5928
L = 2048
INPUT_DIR = f'../../generated_data/model_progan_L_{L}_p_{p}'
max_n_samples = 100
OUPUT_DIR_figs = 'output_files--calc-gr-all-clusters/fig'
OUPUT_DIR_data = 'output_files--calc-gr-all-clusters/txt'

clustering_sample_images = False
calc_stat_of_real_imgs = True
calc_stat_of_fake_imgs = False

## IMPORT MODULES
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import geometric_measure
import plot_func
from modules import gen_class

# AUXILIARY FUNCTIONS
def make_perc_func(L, p):
    def F():
        return (np.random.random(size=(L,L)) < p).astype(int)
    return F

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
                    xlow = 1e1, xup = 1e3, slope_st = '\\tau', show_legend=True, 
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
                    show_slope=True, xlow=2, xup=15, slope_st='\\eta' ,
                    marker='.', markersize=None, show_legend=True, 
                    outfilename=f'{OUPUT_DIR_figs}/gr_{suffix}({datainfo}).pdf')
    plt.close()

## MAIN PART
if __name__ == '__main__':

    print(40*'-')
    filelist = glob.glob(INPUT_DIR + '/' + f'fake_L={L}_p={p}_*.npy')
    if len(filelist) == 0:
        print('There is no file in dir ', INPUT_DIR)
    os.makedirs(OUPUT_DIR_figs, exist_ok=True)
    os.makedirs(OUPUT_DIR_data, exist_ok=True)
    print (f'# L={L} p={p} max_n_samples={max_n_samples}')
    print (f'# out_dir_figs={OUPUT_DIR_figs}, out_dir_data={OUPUT_DIR_data}')
    print(40*'-')


    # CLUSTERING THE TEST IMAGES
    if clustering_sample_images:
        print ('Clustering some of real/fake images ...')
        n_samples = 5
        np.random.seed(72)
        imgs_real = [ make_perc_func(L, p)() for i in range(n_samples) ]

        # now plot some samples
        plt.figure()
        labels_real, _ = geometric_measure.clustering(imgs_real, lower_size=5)
        plot_func.plot_imgs_labels(plt, imgs_real, labels_real, outfilename=f'{OUPUT_DIR_figs}/imgs_real(L={L}).pdf')
        plt.close()

        if filelist and len(filelist) > 0:
            imgs_fake = [np.load(path) for path in filelist[:n_samples]]
            plt.figure()
            labels_fake, _ = geometric_measure.clustering(imgs_fake, lower_size=5)
            plot_func.plot_imgs_labels(plt, imgs_fake, labels_fake, outfilename=f'{OUPUT_DIR_figs}/imgs_fake(L={L}).pdf')
            plt.close()


    # DO STATISTICS
    if calc_stat_of_real_imgs:
        print ('Doing calculations on the real images ...')
        np.random.seed(72)
        img_gen = gen_class.GenUsingFunc(make_perc_func(L, p), max_n_samples)
        do_all_statistics_jobs(img_gen, img_shape=(L,L), suffix='real')

    if calc_stat_of_fake_imgs:
        if filelist and len(filelist) > 0:
            print ('Doing calculations on the fake images ...')
            img_gen = gen_class.GenUsingFile(filelist, max_n_samples)
            do_all_statistics_jobs(img_gen, img_shape=(L,L), suffix='fake')



