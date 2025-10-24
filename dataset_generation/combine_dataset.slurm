#!/bin/bash
#SBATCH --job-name=combine_dataset
#SBATCH --output=.slurm/combine_dataset.out
#SBATCH --error=.slurm/combine_dataset.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Run script
python dataset_generation/combine_dataset.py
