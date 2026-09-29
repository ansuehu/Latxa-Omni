from datasets import load_from_disk

data = load_from_disk("/scratch/asudupe/datasets/VoiceAssistant-400K_eu_norm_v2")

print(data['train'][3])
