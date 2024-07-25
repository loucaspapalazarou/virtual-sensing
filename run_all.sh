#!/bin/bash

sbatch submit.slurm transformer
sbatch submit.slurm mamba
sbatch submit.slurm rnn

