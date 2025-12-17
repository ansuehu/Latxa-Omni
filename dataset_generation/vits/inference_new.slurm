#!/bin/sh

#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=ahotsak_sortu
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=4 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=128GB # RAM memory needed (8-16GB)
#SBATCH --time=2-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --array=0-3
#SBATCH --output=/home/asudupe/Latxa-Omni/.slurm/ahotsak_sortu_backup-%a.out.log
#SBATCH --error=/home/asudupe/Latxa-Omni/.slurm/ahotsak_sortu_backup-%a.err.log

PART_ID=${SLURM_ARRAY_TASK_ID}

python inference_andoni.py --id $PART_ID