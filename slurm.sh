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

python src/hydra/train.py \
    --lattice_size 128 \
    --dataset_size 20 \
    --batch_size 5 \
    --epochs 10 \
    --n_conv_cells 3 \
    --n_convt_cells 5 \
    --bins_number 100 \
    --generator_learning_rate 10e-3 \
    --discriminator_learning_rate 10e-3 \
    --regularization_strength 1.0 \
    --hydra_ratio_bce 1.0 \
    --hydra_ratio_cnn 1.0 \
    --patience_generator 2 \
    --noise_dim 100 \
    --wanted_p 0.5928 \
    --save_dir "./saved_models/hydra" \
    --CNN_model_path "./saved_models/cnn_regression/2021.10.19.11.09.07/model/final_model.pt" \
    --device "cpu" \
    --no-set_generate_plots