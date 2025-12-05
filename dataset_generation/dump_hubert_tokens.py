from datasets import load_from_disk
import torch
import torchaudio
import numpy as np
import joblib
from transformers import Wav2Vec2Processor, HubertModel
import os

TOKEN_DIR = "/scratch/asudupe/datasets/VoiceAssistant-400K_eu/tokens"
os.makedirs(TOKEN_DIR, exist_ok=True)

def extract_features(waveform, model, processor):
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states[9].squeeze(0).cpu().numpy()

def assign_tokens(waveform, model, processor, kmeans):
    features = extract_features(waveform, model, processor)
    tokens = kmeans.predict(features)
    return tokens.tolist()

def get_range_dir(global_idx):
    # size of each range
    block = 10000
    start = (global_idx // block) * block
    end = start + block
    return f"{start}_{end}"

def process_example(example, model, processor, kmeans):
    try:
        audio_path = os.path.join(
            "/scratch/asudupe/datasets/VoiceAssistant-400K_eu",
            example["answer_audio"]
        )

        waveform, sr = torchaudio.load(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)

        tokens = assign_tokens(waveform, model, processor, kmeans)

        # Compute global index from filename
        global_idx = int(os.path.splitext(example["answer_audio"])[0].split("_")[-1])
        filename = f"{global_idx:06d}.npy"

        # Determine output subfolder based on range
        range_dir = get_range_dir(global_idx)
        out_dir = os.path.join(TOKEN_DIR, range_dir)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, filename)
        np.save(out_path, tokens)
        example["answer_token"] = out_path
    except:
        example["answer_token"] = 'error'
    return example

def main():
    model_name = "Ansu/mHubert-basque-ASR"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    model.cuda()

    kmeans = joblib.load("/scratch/asudupe/models/kmeans/basque_hubert_k1000_L9.pt")

    ds = load_from_disk("/scratch/asudupe/datasets/VoiceAssistant-400K_eu/dataset")

    def wrapper(example):
        return process_example(example, model, processor, kmeans)

    ds = ds.map(
        wrapper,
        batched=False,                         # process one example at a time
        desc="Extracting HuBERT tokens & saving to .npy"
    )

    ds.save_to_disk("/scratch/asudupe/datasets/VoiceAssistant-400K_eu/dataset_with_token_paths")

if __name__ == "__main__":
    main()

