#!/bin/bash
#SBATCH --account=rwth1651
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1


module load CUDA
module load Python
module load PyTorch

apptainer exec --nv $PYTORCH_IMAGE python3 /path/to/your/script.py