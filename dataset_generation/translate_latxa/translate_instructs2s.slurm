#!/usr/bin/env bash
# Run `latxa-txat` (activates the latxa-txat conda env) in your shell before
# `sbatch`-ing this -- sbatch inherits the submitting shell's environment,
# matching the convention used by every other job in this repo (see
# AGENTS.md). vLLM and its dependencies (transformers, xgrammar, ...) are
# installed there.
#
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=translate-instructs2s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andoni.sudupe@ehu.eus
#SBATCH --output=.slurm/translate-instructs2s.%A_%a.out.log
#SBATCH --error=.slurm/translate-instructs2s.%A_%a.err.log
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

MODEL_PATH=HiTZ/Latxa-Llama-3.1-70B-Instruct
JSONL_PATH=/scratch/asudupe/datasets/InstructS2S-200K/instruct_en_200k.jsonl
OUTPUT_PATH=/scratch/asudupe/datasets/InstructS2S-200K/instruct_en_200k_eu

python translate_instructs2s.py \
    --model_path "$MODEL_PATH" \
    --jsonl_path "$JSONL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --tensor_parallel_size 2 \
    --batch_size 128 \
    --shard_idx "$SLURM_ARRAY_TASK_ID" \
    --num_shards "$SLURM_ARRAY_TASK_COUNT"
