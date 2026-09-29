#!/usr/bin/env bash
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --job-name=normalize-va
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andoni.sudupe@ehu.eus
#SBATCH --output=.slurm/normalize-va.out.log
#SBATCH --error=.slurm/normalize-va.err.log
#SBATCH --chdir=/home/asudupe/Latxa-Omni/dataset_generation

export HF_DATASETS_CACHE=/scratch/asudupe/hf_cache
export HF_HOME=/scratch/asudupe/hf_home

mkdir -p $HF_DATASETS_CACHE $HF_HOME

python normalize_voice_assistant.py --validate --num_proc 8 --dataset_path /scratch/asudupe/datasets/VoiceAssistant-400K_eu --dataset_out /scratch/asudupe/datasets/VoiceAssistant-400K_eu_norm_v2
