#!/bin/bash

# Set variables
FEAT_DIR="speechocean/speech_units/hubert_features"            # 存放 .npy/.len 的資料夾
KM_PATH="models/wav2unit/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"  # K-means 模型路徑
NSHARD=1                       # 分幾 shard 跑，通常設 1 即可
RANK=0                         # 第幾個 shard（從 0 開始）
LAB_DIR="speechocean/speech_units/km_labels"            # 輸出 .km 檔案的資料夾

# 要處理的 splits
for SPLIT in train valid test; do
  echo "Processing $SPLIT..."
  python fairseq/examples/hubert/simple_kmeans/dump_km_label.py $FEAT_DIR $SPLIT $KM_PATH $NSHARD $RANK $LAB_DIR
done
