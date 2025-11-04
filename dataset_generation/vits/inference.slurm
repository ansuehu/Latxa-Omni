#!/bin/bash
START=$1
END=$2
DATA_PATH=$3
MODEL_PATH=$4
python inference_andoni.py \
    --start $START \
    --end $END \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH