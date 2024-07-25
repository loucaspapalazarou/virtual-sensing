#!/bin/bash

sbatch submit.slurm transformer fast_dev_run
sbatch submit.slurm mamba fast_dev_run
sbatch submit.slurm rnn fast_dev_run

