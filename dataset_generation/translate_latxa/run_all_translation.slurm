#!/usr/bin/env bash

SLURM_DIR=$(pwd)"/.slurm"
cd dataset_generation/translate_latxa
SPLITS=(
    "0 100000"
    "100000 200000"
    "200000 300000"
    "300000 500000"
    # "400000 500000"
)


MODEL_PATH=HiTZ/Latxa-Llama-3.1-70B-Instruct
DATA_PATH=/scratch/asudupe/datasets/VoiceAssistant-400K
PROMPT_PATH=prompt.j2
FREQUENCY_PENALTY=0.15
TENSOR_PARALLEL_SIZE=2
BATCH_SIZE=256

for i in "${!SPLITS[@]}"; do
    SPLIT=(${SPLITS[$i]})
    echo "Starting translation for split: ${SPLIT}"
    SLURM_ARGS=$(
        echo \
            --partition=hitz-exclusive \
            --account=hitz-exclusive \
            --job-name="dataset_translation_${i}" \
            --mem=256G \
            --gres=gpu:2\
            --cpus-per-task=2 \
            --constraint=a100-sxm4 \
        --time=2-00:00:00 \
            --mail-user=andoni.sudupe@ehu.eus \
            --output="${SLURM_DIR}/dataset_translation_${SPLIT[0]}_${SPLIT[1]}.out" \
            --error="${SLURM_DIR}/dataset_translation_${SPLIT[0]}_${SPLIT[1]}.err"
    )
    echo "SLURM_ARGS: ${SLURM_ARGS}"
    sbatch ${SLURM_ARGS} run_single_translation.slurm \
        $MODEL_PATH \
        $DATA_PATH \
        $PROMPT_PATH \
        $FREQUENCY_PENALTY \
        $TENSOR_PARALLEL_SIZE \
        ${SPLIT[0]} \
        ${SPLIT[1]} \
        $BATCH_SIZE
done