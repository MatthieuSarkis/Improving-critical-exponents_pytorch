#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH -J dat-1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=32GB
#SBATCH --job-name="diffusion_ising"
#SBATCH --output="OUTPUT_factory_ising.out" # job standard output file (%j replaced by job id)
#SBATCH --error="ERROR_factory_ising.out" # job standard error file (%j replaced by job id)

ulimit -s unlimited
export OMP_NUM_THREADS=1
#export MODULEPATH=/opt/apps/resif/iris/2019b/broadwell/modules/all/
module load lang/Python/3.8.6-GCCcore-10.2.0
. /home/users/msarkis/git_repositories/Improving-critical-exponents_pytorch/.env/bin/activate
module load toolchain/intel

python src/denoising-diffusion-pytorch/sample.py

#python $1