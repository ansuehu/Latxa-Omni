#!/usr/bin/env bash
# One-off remediation run. The first full pass (array job 7403402) left the
# target word in English inside the *request* turn of roughly half of the
# "make a sentence using the words X, Y, Z" conversations -- e.g.
# "sortu ... purchase, online eta store hitzak erabiliz?" -- even though the
# answer turn correctly used the Basque equivalents. Guideline 9 and a
# matching few-shot example were added to translate_instructs2s.py to fix
# this; this job re-translates only the affected conversations (selected by
# the same regex used to build vocab_target_subset.jsonl), with the same 70B
# model as the main run so the corpus stays internally consistent.
#
# Run `latxa-txat` in your shell before `sbatch`-ing this.
#
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=retranslate-vocab-targets
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andoni.sudupe@ehu.eus
#SBATCH --output=.slurm/retranslate-vocab-targets.%j.out.log
#SBATCH --error=.slurm/retranslate-vocab-targets.%j.err.log
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
    --model_path HiTZ/Latxa-Llama-3.1-70B-Instruct \
    --jsonl_path /scratch/asudupe/datasets/InstructS2S-200K/vocab_target_subset.jsonl \
    --output_path /scratch/asudupe/datasets/InstructS2S-200K/vocab_target_subset_eu \
    --tensor_parallel_size 2 \
    --batch_size 128
