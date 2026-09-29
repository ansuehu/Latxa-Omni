#!/bin/bash
MODEL_PATH=$1
DATASET_PATH=$2
FREQUENCY_PENALTY=$3
TENSOR_PARALLEL_SIZE=$4
DATASET_START=$5
DATASET_END=$6
BATCH_SIZE=$7
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"
python translate.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --frequency_penalty $FREQUENCY_PENALTY \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --dataset_start $DATASET_START \
    --dataset_end $DATASET_END \
    --batch_size $BATCH_SIZE
