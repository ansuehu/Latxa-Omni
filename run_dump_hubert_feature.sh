#!/bin/bash

# ========== CONFIG ==========
TSV_DIR="librispeech/data"                   # 包含 .tsv 的資料夾
CKPT_PATH="models/wav2unit/mhubert_base_vp_en_es_fr_it3.pt"       # HuBERT checkpoint 路徑
LAYER=11                                # 特徵抽取的 transformer 層
NSHARD=1                                # 分成幾份處理（單 GPU 可設為 1）
RANK=0                                  # 該份的編號（單 GPU 可設為 0）
FEAT_DIR="librispeech/data/speech_units/hubert_features"          # 特徵輸出資料夾

# ========== RUN ==========
for SPLIT in train dev test
do
    echo "Processing split: $SPLIT"
    python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py \
        "$TSV_DIR" \
        "$SPLIT" \
        "$CKPT_PATH" \
        "$LAYER" \
        "$NSHARD" \
        "$RANK" \
        "$FEAT_DIR"
done

echo "All splits processed successfully."
