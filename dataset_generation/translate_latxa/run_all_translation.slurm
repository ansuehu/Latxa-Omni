#! /bin/bash
SLURM_DIR=$(pwd)"/.slurm"
cd dataset_generation/translate_latxa
SPLITS=(
    "0 500000"
)
for i in "${!SPLITS[@]}"; do
    SPLIT=(${SPLITS[$i]})
    echo "Starting translation for split: ${SPLIT}"
    SLURM_ARGS=$(
        echo \
            --partition=hitz-exclusive \
            --account=hitz-exclusive \
            --job-name="dataset_translation_${i}" \
            --mem=32G \
            --gres=gpu:1 \
            --constraint=a100 \
            --cpus-per-task=4 \
            --constraint=a100-sxm4 \
        --time=1-00:00:00 \
            --mail-user=andoni.sudupe@ehu.eus \
            --output="${SLURM_DIR}/dataset_translation_${SPLIT[0]}_${SPLIT[1]}.out" \
            --error="${SLURM_DIR}/dataset_translation_${SPLIT[0]}_${SPLIT[1]}.err"
    )
    echo "SLURM_ARGS: ${SLURM_ARGS}"
    sbatch ${SLURM_ARGS} run_single_translation.slurm \
        $DATA/models/Latxa3.1_8b_lr1e-5 \
        HiTZ/Magpie-Llama-3.1-8B-Instruct-Filtered \
        prompt.j2 \
        0.15 \
        1 \
        ${SPLIT[0]} \
        ${SPLIT[1]}
done
