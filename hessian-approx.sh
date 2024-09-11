#!/bin/bash
#SBATCH --partition=A40medium 
#SBATCH --time=23:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
module load CUDA
module load Python
module load PyTorch


python3 hessian-approx/hessian-approx.py