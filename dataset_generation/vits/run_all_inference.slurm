#!/usr/bin/env bash
SLURM_DIR=$(pwd)"/.slurm"
cd /home/asudupe/Latxa-Omni/dataset_generation/vits
SPLITS=(
    "0 100000"
    "100000 200000"
    "200000 300000"
    "300000 500000"
)

DATA_PATH=/scratch/asudupe/datasets/translation/
MODEL_PATH=/scratch/asudupe/models/vits/

latxa-txat

# for i in "${!SPLITS[@]}"; do
for i in 0; do
    SPLIT=(${SPLITS[$i]})
    echo "Starting translation for split: ${SPLIT}"
    SLURM_ARGS=$(
        echo \
            --partition=hitz-exclusive \
            --account=hitz-exclusive \
            --job-name="speech_synthesis_${i}" \
            --mem=64G \
            --gres=gpu:1\
            --cpus-per-gpu=24 \
            --constraint=a100-sxm4 \
        --time=2-00:00:00 \
            --mail-user=andoni.sudupe@ehu.eus \
            --output="${SLURM_DIR}/speech_synthesis_${SPLIT[0]}_${SPLIT[1]}.out" \
            --error="${SLURM_DIR}/speech_synthesis_${SPLIT[0]}_${SPLIT[1]}.err"
    )
    echo "SLURM_ARGS: ${SLURM_ARGS}"
    sbatch ${SLURM_ARGS} inference.slurm \
        ${SPLIT[0]} \
        ${SPLIT[1]} \
        $DATA_PATH \
        $MODEL_PATH
done