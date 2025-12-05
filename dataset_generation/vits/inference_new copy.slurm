#!/bin/sh

#SBATCH --qos=regular
#SBATCH --constraint=a100-pcie
#SBATCH --job-name=ahotsak_sortu
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=64GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=.slurm/ahotsak_sortu.out.log
#SBATCH --error=.slurm/ahotsak_sortu.err.log
#SBATCH --chdir=/home/asudupe/Latxa-Omni


python dataset_generation/vits/inference_andoni.py