#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=72:00:00
#SBATCH --account=tc064-s2567498
#SBATCH --gres=gpu:1

# Set environment variables for writable directories
export MPLCONFIGDIR=/work/tc064/tc064/s2567498/matplotlib
export HF_HOME=/work/tc064/tc064/s2567498/transformers_cache
export TORCH_HOME=/work/tc064/tc064/s2567498/torch_home

# Load the required modules
source /work/tc064/tc064/s2567498/venv/bin/activate

cd /work/tc064/tc064/s2567498/msc-diss/utils

srun python frame_reduction.py
