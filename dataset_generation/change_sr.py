import os
import numpy as np
from scipy.signal import resample_poly
from datasets import load_dataset

# ----- CONFIG -----
REPO_NAME = "Ansu/Instruct_S2S_eu"  # dataset repo name
OUTPUT_DIR = "resampled_splits"     # folder to save splits
NUM_SPLITS = 4                      # same as --array=0-3
# -------------------

# Slurm task ID
idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

def resample_audio(example):
    audio = np.array(example["question_audio"], dtype=np.float32)
    example["question_audio"] = resample_poly(audio, 16000, 22050).tolist()
    audio = np.array(example["answer_audio"], dtype=np.float32)
    example["answer_audio"] = resample_poly(audio, 16000, 22050).tolist()
    return example

# Load dataset
dataset = load_dataset(REPO_NAME, split="train")

# Split dataset into shards
split = dataset.shard(num_shards=NUM_SPLITS, index=idx)

# Apply transformation
split = split.map(resample_audio, num_proc=8, desc=f"Resampling split {idx}")

# Save split
os.makedirs(OUTPUT_DIR, exist_ok=True)
split.save_to_disk(f"{OUTPUT_DIR}/split_{idx}")
print(f"✅ Finished resampling split {idx}")
