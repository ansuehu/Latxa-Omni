#!/bin/bash
#SBATCH --job-name=resample_audio
#SBATCH --output=.slurm/resample_audio_%A_%a.out
#SBATCH --error=.slurm/resample_audio_%A_%a.err
#SBATCH --array=0-5              # 4 splits (0,1,2,3)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Run script
python dataset_generation/change_sr.py $SLURM_ARRAY_TASK_ID
