from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

splits = [load_from_disk(f"resampled_splits/split_{i}") for i in range(4)]
merged = concatenate_datasets(splits)

remove_indices = []
for i, a in enumerate(tqdm(merged['question_audio'])):
    if not any(a):
        remove_indices.append(i)
        print(f'removed {i}')

merged = merged.select(
    (
        i for i in range(len(merged)) 
        if i not in set(remove_indices)
    )
)
print(f"✅ Removed {len(remove_indices)} empty audio samples")
merged.save_to_disk("/scratch/andoni.sudupe/Instruct_S2S_eu/hf_16k")
print("✅ Merged all splits into '/scratch/andoni.sudupe/Instruct_S2S_eu/hf_16k'")