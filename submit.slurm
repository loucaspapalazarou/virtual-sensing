#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --account=tc064-s2567498
#SBATCH --gres=gpu:2

# Check if the model argument is provided
if [ -z "$1" ]; then
  echo "No model specified. Usage: sbatch train.sh <model_name> [fast_dev_run]"
  exit 1
fi

MODEL=$1

# Check if the second argument is provided
if [ ! -z "$2" ] && [ "$2" == "fast_dev_run" ]; then
  FAST_DEV_RUN="--fast-dev-run"
else
  FAST_DEV_RUN=""
fi

# Set environment variables for writable directories
export MPLCONFIGDIR=/work/tc064/tc064/s2567498/matplotlib
export HF_HOME=/work/tc064/tc064/s2567498/transformers_cache
export TORCH_HOME=/work/tc064/tc064/s2567498/torch_home

export SLURM_NTASKS_PER_NODE=2
export SLURM_NTASKS=2

# Load the required modules
source /work/tc064/tc064/s2567498/venv/bin/activate

cd /work/tc064/tc064/s2567498/virtual-sensing

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE python train.py --model $MODEL $FAST_DEV_RUN
