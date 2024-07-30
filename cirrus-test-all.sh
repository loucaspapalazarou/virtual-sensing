#!/bin/bash

sbatch submit-cirrus.slurm transformer fast_dev_run
sbatch submit-cirrus.slurm mamba fast_dev_run
sbatch submit-cirrus.slurm rnn fast_dev_run

