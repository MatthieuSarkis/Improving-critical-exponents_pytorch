# %% [markdown]
# ## Initial Parameters

# %%
INPUT_DIR = '../../data/generated_data/model_progan_2022.05.13.13.43.18'
p = 0.5928
L = 1024
max_n_samples = 100
OUPUT_DIR = 'out_stat'

clustering_sample_images = True
calc_stat_of_real_imgs = True
calc_stat_of_fake_imgs = False

# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import geometric_measure
import fit 
import gen_class

filelist = glob.glob(INPUT_DIR + '/' + f'fake_L={L}_p={p}_*.npy')
if len(filelist) == 0:
    print('There is no file in dir ', INPUT_DIR)
os.makedirs(OUPUT_DIR, exist_ok=True)
print (f'# L={L} p={p} max_n_samples={max_n_samples}')
# %% [markdown]
# # Clustering the Input Images

# %%
def plot_imgs_labels(imgs, labels, outfilename):
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

if clustering_sample_images:
    print ('# Clustering some of real/fake images')
    n_samples = 5
    np.random.seed(72)
    imgs_real = [ (np.random.random(size=(L,L)) < p).astype(int) for i in range(n_samples) ]

    # now plot some samples
    plt.figure(1)
    labels_real, _ = geometric_measure.clustering(imgs_real, lower_size=5)
    plot_imgs_labels(imgs_real, labels_real, outfilename=f'{OUPUT_DIR}/imgs_real(L={L}).pdf')
    plt.close()

    if filelist and len(filelist) > 0:
        imgs_fake = [np.load(path) for path in filelist[:n_samples]]
        plt.figure(2)
        labels_fake, _ = geometric_measure.clustering(imgs_fake, lower_size=5)
        plot_imgs_labels(imgs_fake, labels_fake, outfilename=f'{OUPUT_DIR}/imgs_fake(L={L}).pdf')
        plt.close()

# %% [markdown]
# # Statistics

# %%

def logplotXY(plt, x, y, xlabel, ylabel, title=None, outfilename=None,
              xlow = 1e1, xup = 1e3,
              slope_st = '\\tau',):
              
    plt.loglog(x, y, ls='', marker='o', fillstyle='none', 
               markersize = 5,
               label = 'sim')

    # plot slope
    indx = (x >= xlow) & (x <= xup)
    x, y = x[indx], y[indx]
    expo, c, expo_err, c_err = fit.loglog_slope(x, y)
    xn = np.logspace(np.log10(x[0]), np.log10(x[-1]))
    yn = c * xn ** expo
    expo_usign = expo if expo > 0 else -expo
    plt.loglog(xn, yn, color='k', lw=1,
               label = fr'${slope_st}={expo_usign:.2f} \pm {expo_err:.2f}$'  )
    ###
    plt.legend(frameon=False)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel, fontsize=16)
    if ylabel:
        plt.ylabel(ylabel, fontsize=16)
    if outfilename:
        plt.savefig(outfilename, pad_inches=0.01, bbox_inches='tight')

# %% [markdown]
# ### Real images

# %%
def make_perc_arr(L, p):
    def F():
        return (np.random.random(size=(L,L)) < p).astype(int)
    return F

# %%
if calc_stat_of_real_imgs:
    print ('# Doing calculations on the real images ...')
    np.random.seed(72)
    img_gen_real = gen_class.GenUsingFunc(make_perc_arr(L, p), max_n_samples)
    n_samples = img_gen_real.len()
    # get the mesures related to the configurations
    measure_real = geometric_measure.get_measure(img_gen_real, img_shape=(L, L))
    # get the statistics of measures
    stat_real = geometric_measure.measure_statistics(measure_real, nbins_for_ns=43)

    # %%
    ns = stat_real['ns']
    x, y, dx = ns['bin_centers'], ns['hist'], ns['bin_sizes']
    plt.figure(4)
    logplotXY(plt, x, y, '$s$', '$n(s)$', xlow = 1e1, xup = 1e3, slope_st = '\\tau',
            outfilename = f'{OUPUT_DIR}/ns_real(L={L},N={n_samples}).pdf',
            )
    plt.close()

# %% [markdown]
# ### Fake images

# %%
if calc_stat_of_fake_imgs:
    print ('# Doing calculations on the fake images ...')
    img_gen_fake = gen_class.GenUsingFile(filelist, max_n_samples)
    n_samples = img_gen_fake.len()
    measure_fake = geometric_measure.get_measure(img_gen_fake, img_shape=(L, L))
    stat_fake = geometric_measure.measure_statistics(measure_fake, nbins_for_ns=43)

    # %%
    ns = stat_fake['ns']
    x, y, dx = ns['bin_centers'], ns['hist'], ns['bin_sizes']
    plt.figure(5)
    logplotXY(plt, x, y, '$s$', '$n(s)$', xlow = 1e1, xup = 1e3, slope_st = '\\tau',
            outfilename = f'{OUPUT_DIR}/ns_fake(L={L},N={n_samples}).pdf',
            )
    plt.close()

# %%



