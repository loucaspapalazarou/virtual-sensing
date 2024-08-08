#!/bin/bash

sbatch submit-cirrus.slurm transformer
sbatch submit-cirrus.slurm mamba
sbatch submit-cirrus.slurm rnn

