#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH -J dat-1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -N 1             # request 1 nodes
#SBATCH -n 1     # allocate one task per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=32GB

ulimit -s unlimited
export OMP_NUM_THREADS=1
#export MODULEPATH=/opt/apps/resif/iris/2019b/broadwell/modules/all/
module load lang/Python
. /home/users/msarkis/git_repositories/Improving-critical-exponents_pytorch/.env/bin/activate
module load toolchain/intel
python src/main.py

#python $1