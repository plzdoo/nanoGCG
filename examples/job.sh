#!/bin/sh -l

#SBATCH --mail-user=liang328@purdue.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -A antoniob-k
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1 --mem=456G
#SBATCH --ntasks=1 --cpus-per-task=8

python simple.py