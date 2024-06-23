#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=06:00:00
#SBATCH --gpus=2
#SBATCH --ntasks=1
module load CUDA
module load Python
module load PyTorch

python train.py