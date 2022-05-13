#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH -J dat-1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --nodes=2   # number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=32GB
#SBATCH --job-name="ProGAN"
#SBATCH --output="OUTPUT.out" # job standard output file (%j replaced by job id)
#SBATCH --error="ERROR.out" # job standard error file (%j replaced by job id)

ulimit -s unlimited
export OMP_NUM_THREADS=1
#export MODULEPATH=/opt/apps/resif/iris/2019b/broadwell/modules/all/
module load lang/Python
. /home/users/msarkis/git_repositories/Improving-critical-exponents_pytorch/.env/bin/activate
module load toolchain/intel

python src/progan/main.py
#python src/progan/factory.py

#python $1