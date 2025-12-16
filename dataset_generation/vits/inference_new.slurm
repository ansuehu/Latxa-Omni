#!/bin/sh

#SBATCH --partition=gpu-H100
#SBATCH --job-name=ahotsak_sortu
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=64GB # RAM memory needed (8-16GB)
#SBATCH --time=1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --array=0-3
#SBATCH --output=.slurm/ahotsak_sortu-%a.out.log
#SBATCH --error=.slurm/ahotsak_sortu-%a.err.log

PART_ID=${SLURM_ARRAY_TASK_ID}

python dataset_generation/vits/inference_andoni.py --id $PART_ID