#!/usr/bin/env bash
# One-off pilot run (verification step 5 of the InstructS2S-eu translation
# plan): 200 conversations, 1 GPU, the 8B model instead of the 70B, no
# sharding. Run `latxa-txat` in your shell before `sbatch`-ing this --
# sbatch inherits the submitting shell's environment (see AGENTS.md).
#
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=translate-instructs2s-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andoni.sudupe@ehu.eus
#SBATCH --output=.slurm/translate-instructs2s-pilot.%j.out.log
#SBATCH --chdir=/home/asudupe/Latxa-Omni/dataset_generation/translate_latxa

export HF_DATASETS_CACHE=/scratch/asudupe/hf_cache
export HF_HOME=/scratch/asudupe/hf_home
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python translate_instructs2s.py \
    --model_path HiTZ/Latxa-Llama-3.1-8B-Instruct \
    --jsonl_path /scratch/asudupe/datasets/InstructS2S-200K/instruct_en_200k.jsonl \
    --output_path /scratch/asudupe/datasets/InstructS2S-200K/pilot/instruct_en_200k_eu_pilot \
    --tensor_parallel_size 1 \
    --batch_size 50 \
    --limit 200
