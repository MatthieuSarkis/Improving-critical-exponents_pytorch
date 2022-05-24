import glob, os
import numpy as np
import matplotlib.pyplot as plt
import plot_func
#================================
basename = 'ns'
p = 0.5928
L_arr = [128, 256, 512, 1024, 2048]
INPUT_DIR = 'output_files/data'
OUPUT_DIR = 'output_files/figs_distributions'
#================================
dic_xlabel = {'gr': '$r$', 'ns': '$s$'}
dic_ylabel = {'gr': '$g(r)$', 'ns': '$n(s)$'}
#dict_bounds = {'gr': {'128': ()}}

if __name__ == '__main__':

    plt.figure()
    os.makedirs(OUPUT_DIR, exist_ok=True)
    

    for L in L_arr:

        for suffix in ['real', 'fake']:
            filename_pattern = f'{basename}_{suffix}(p={p},L={L},*).dat'
            fileslist = glob.glob(os.path.join(INPUT_DIR, filename_pattern))
            if len(fileslist) == 0:
                continue
            print(f'L={L}, suffix={suffix}')
            
            data = np.loadtxt(fileslist[0])
            x, y = data[:,0], data[:,1]
            
            if basename == 'gr':
                plot_func.logplotXY(plt, x, y, 
                                    sim_st=fr'$L={L} \quad {suffix}$',
                                    scale_xy_logplot= 1.01,
                                    show_slope=True, xlow=2, xup=15, slope_st='\\eta' ,
                                    marker='.', markersize=None, )
            elif basename == 'ns':
                plot_func.logplotXY(plt, x, y,
                                    sim_st=fr'$L={L} \quad {suffix}$',
                                    xlow = 1e1, xup = 1e3, slope_st = '\\tau', )


    outfilename = f'{OUPUT_DIR}/{basename}(p={p}).pdf'
    xlabel, ylabel = dic_xlabel[basename], dic_ylabel[basename]
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(frameon=False)
    plt.savefig(outfilename, pad_inches=0.01, bbox_inches='tight')