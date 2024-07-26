#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --account=tc064-s2567498
#SBATCH --gres=gpu:1

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

# Load environment variables from .bashrc
source ~/.bashrc

# Ensure environment variables are set
: "${WORK_DIR:?}"
: "${VENV_DIR:?}"
: "${PROJECT_DIR:?}"
: "${MPLCONFIGDIR:?}"
: "${HF_HOME:?}"
: "${TORCH_HOME:?}"

export SLURM_NTASKS_PER_NODE=1
export SLURM_NTASKS=1

# Load the required modules
source $VENV_DIR/bin/activate

cd $PROJECT_DIR

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE python train.py --model $MODEL $FAST_DEV_RUN
