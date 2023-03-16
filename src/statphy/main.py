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
p = 0.5927
L = 1024
idir_fake = f'./generated_data/fake-denoising-diffusion-pytorch/perc/2023.03.06.23.10.08'
idir_real = f'./generated_data/real/perc'
max_n_samples = 5000
odir = 'output_files-perc'

clustering_sample_images = True
calc_statistics = True

######################## IMPORT MODULES #########################
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import geometric_measure
import plt_funcs
from modules import gen_class


######################## USEFULL FUNCTIONS ######################

def get_data(idir, sep='_') -> np.ndarray:
    """
    read all files matched with ext and append the data, and return it
    """
    filepattern = f'*L{sep}{L}_p{sep}{p}*.*'
    filelist = glob.glob(os.path.join(idir, filepattern))

    if len(filelist) == 0:
        return None

    data = np.empty((0,L,L), np.int8)

    for path in filelist:

        filename, ext = os.path.splitext(path)

        if ext == '.npy':
            curr = np.load(path).astype(np.int8)
        else:
            curr = np.fromfile(path, np.int8).reshape((-1,L,L))

        data = np.append(data, curr, axis=0)
    
    return data

######################## MAIN FUNCTION ######################
if __name__ == '__main__':


    def do_all_jobs(img_gen, suffix='real'):

        n_samples = len(img_gen)

        # get the statistics of measures
        measure_obj = geometric_measure.Measure()

        history = measure_obj.calc_history(img_gen, (L,L), 1, True)
        stats = measure_obj.get_stats(nbins_for_ns=33, nbins_for_m_gr=33)


        filenamesuffix = f'(L={L},p={p},n={n_samples})--{suffix}'
        plt_funcs.plot_stats(plt, stats, odir_figs, filenamesuffix, f'$L={L}$' )
   
    print(50*'-')
    print(f'idir_real: {idir_real}')
    print(f'idir_fake: {idir_fake}')
    
    realdata = get_data(idir_real,)
    fakedata = get_data(idir_fake,)


    if fakedata is None:
        print(f'No fakedata!')
    else:
        print(f'fakedata.shape={fakedata.shape}')

    if realdata is None:
        print(f'No realdata!')
    else:
        print(f'realdata.shape={realdata.shape}')

   
    odir_figs, odir_txt = f'{odir}/fig', f'{odir}/txt'
    os.makedirs(odir_figs, exist_ok=True)
    os.makedirs(odir_txt, exist_ok=True)
    
    print(50*'-')
 


    #### CLUSTERING SOME OF IMAGES ###
    if clustering_sample_images:
        print ('Clustering some of real/fake images ...')
        n_samples = 5

        if fakedata is not None:
            imgs = fakedata[:n_samples]
            plt.figure()
            labels, _ = geometric_measure.clustering(imgs, lower_size=5)
            plt_funcs.plot_imgs_labels(plt, imgs, labels, outfilename=f'{odir_figs}/fake_imgs(L={L},p={p}).pdf')
            plt.close()

        if realdata is not None:
            imgs = realdata[:n_samples]
            plt.figure()
            labels, _ = geometric_measure.clustering(imgs, lower_size=5)
            plt_funcs.plot_imgs_labels(plt, imgs, labels, outfilename=f'{odir_figs}/real_imgs(L={L},p={p}).pdf')
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

