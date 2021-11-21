#!/bin/bash

#SBATCH --job-name="hydra"
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --mem=20G   # maximum memory per node
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --output=hydra.out # output file

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

#export HDF5_USE_FILE_LOCKING='FALSE'  # for exporting hd5 file
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

source .env/bin/activate # activate your environment

python src/statphy/data_factory.py      \
	--model square_lattice_percolation  \
	--L 128                          \
	--control_parameter 0.5928        \
	--samples 50000                      \
	--path "./data/simulation-50k"

python src/GAN/train.py \
	--data_path "./data/simulation-50k/L=128_p=0.5928.npz" \
	--batch_size 500 \
	--epochs 1000 \
	--noise_dim 100 \
	--save_dir ./data/models/gan-50k
