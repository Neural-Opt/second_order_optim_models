#!/bin/bash
#SBATCH --partition=A40medium 
#SBATCH --time=22:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1
module load CUDA
module load Python
module load PyTorch


python3 train.py