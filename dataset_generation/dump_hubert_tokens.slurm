#!/usr/bin/env bash
#SBATCH --account=hitz-exclusive 
#SBATCH --partition=hitz-exclusive
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=dump-hubert-token# Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=64GB # RAM memory needed (8-16GB)
#SBATCH --time=4-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/dump-hubert-token.out.log
#SBATCH --error=.slurm/dump-hubert-token.err.log
#SBATCH --chdir=/home/asudupe/Latxa-Omni

python dataset_generation/dump_hubert_tokens.py