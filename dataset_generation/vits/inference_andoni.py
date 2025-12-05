# import matplotlib.pyplot as plt
# import IPython.display as ipd
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import argparse
import os
import json
# import math
import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
import commons
from utils import get_hparams_from_file, load_checkpoint
# from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
import pyximport
pyximport.install()

from models import SynthesizerTrn
from text.symbols import symbols
from text.symbols_cast import symbols_cast
from text import text_to_sequence

from scipy.io.wavfile import write

import soundfile as sf
import speech
import numpy as np
import time
# import sys
from datasets import Dataset, Audio, load_from_disk
import multiprocess
import re
from functools import partial
import librosa
from transformers import Wav2Vec2Processor, HubertModel
import joblib


hps_marina = get_hparams_from_file("/scratch/asudupe/models/vits/configs/sonora.json")

net_g_marina = SynthesizerTrn(
    len(symbols),
    hps_marina.data.filter_length // 2 + 1,
    hps_marina.train.segment_size // hps_marina.data.hop_length,
    **hps_marina.model).cuda()
_ = net_g_marina.eval()

_ = load_checkpoint("/scratch/asudupe/models/vits/checkpoints/marina_898.pth", net_g_marina, None)

# hps_alex = get_hparams_from_file("/scratch/asudupe/models/vits/configs/sonora.json")

# net_g_alex = SynthesizerTrn(
#     len(symbols),
#     hps_alex.data.filter_length // 2 + 1,
#     hps_alex.train.segment_size // hps_alex.data.hop_length,
#     **hps_alex.model).cuda()
# _ = net_g_alex.eval()

# _ = load_checkpoint("/scratch/asudupe/models/vits/checkpoints/alex_864.pth", net_g_alex, None)

# hps_comb = get_hparams_from_file("/scratch/asudupe/models/vits/configs/multispeaker.json")

# net_g_comb = SynthesizerTrn(
#     len(symbols),
#     hps_comb.data.filter_length // 2 + 1,
#     hps_comb.train.segment_size // hps_comb.data.hop_length,
#     n_speakers=9,
#     **hps_comb.model).cuda()
# _ = net_g_comb.eval()

# _ = load_checkpoint("/scratch/asudupe/models/vits/checkpoints/multispeaker_500000.pth", net_g_comb, None)

speakers = {
    0: "Aintzane",
    1: "Jaione",
    2: "Klara",
    3: "Monika",
    4: "Kiko",
    5: "Inaki",
    6: "Kepa",
    7: "Pello",
    8: "Xabier",
    9: "Marina",
    10: "Alex"
}
model_name = "Ansu/mHubert-basque-ASR"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)
model.eval()
model.cuda()

kmeans = joblib.load("/scratch/asudupe/models/kmeans/basque_hubert_k1000_L9.pt")
# hps_kristof = get_hparams_from_file("./configs/kristof_eu.json")

DATASET_PATH='/scratch/asudupe/datasets/VoiceAssistant-400K_eu'

def sanitize(s):
    return "".join(c if ord(c) < 128 else "_" for c in s)

def clean_text(text):
    text = text.replace(':', ',')
    text = text.replace(';', ',')
    text = text.replace('(', ',')
    text = text.replace(')', ',')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("“", '')
    text = text.replace("”", '')
    text = text.replace("ñ", 'n')
    text = text.replace("á", 'a')
    text = text.replace("é", 'e')
    text = text.replace("í", 'i')
    text = text.replace("ó", 'o')
    text = text.replace("ú", 'u')
    # print(output)
    return text

def getPhones(text, language):
    #####################################
    # Extracción fonética de las frases #
    #####################################
    text = text.lstrip()
    text = sanitize(text)
    cleaned_text = clean_text(text)
    #phones = speech.modulo1y2(clean_text, mode='Spell', PhTSimple='y', language=language, keep_chars=None, verbose=True)
    command = f"echo {cleaned_text} | iconv -f UTF-8 -t ISO-8859-1 | ./modulo1y2 -HDic=dict/eu_dic -Lang=eu -TxtMode=Spell -PhTSimple=y 2> /dev/null | iconv -f ISO-8859-1 -t UTF-8"
    phones = os.popen(command).read()
    # print('phones:', phones)

    command = f"echo {cleaned_text} | iconv -f UTF-8 -t ISO-8859-1 | ./modulo1y2 -HDic=dict/eu_dic -Lang=eu -TxtMode=Word -PhTSimple=n 2> /dev/null | iconv -f ISO-8859-1 -t UTF-8"
    checker = os.popen(command).read()
    # print('checker:', checker)
    #checker = speech.modulo1y2(clean_text, mode='Word', PhTSimple='n', language=language, keep_chars=None, verbose=False)
    # phones = phones.replace(" ", "")
    slp_all = []
    for ph, ch in zip(phones.split('\n'), checker.split('\n')):
        clp = ""
        for p in range(len(ph)):
            # print(phones[p])
            if ph[p]=='\n':
                clp = clp + ' | '
            else:
                clp = clp + "".join(ph[p].split('-'))
            # print(f"clp: {clp}, phone: {phones[p]}")
            if p == len(ph) - 1:
                clp = clp + ' | '
        
        slp = str(clp).split()
        # print(slp)
        if '?' in ch:
            slp.append('?')
        elif '!' in ch:
            slp.append('!')
        elif '.' in ch:
            slp.append('.')
        else:
            slp.append('.')

        slp_all.extend(slp)
    phones = np.array(slp_all)

    # print(phones)

    return phones

def get_text(text, hps, language, path=False):
    if not path:
        text = getPhones(text, language)
        # print(f'text: {text}')
    text_norm = text_to_sequence(text, hps.data.text_cleaners, language, inference=not path)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def infer_voice(question, voice, device): 
    sid=0
    if voice <= 8:
        net_g = net_g_comb
        hps = hps_comb
        sid = voice
    elif voice == 9:
        net_g = net_g_marina
        hps = hps_marina 
    elif voice == 10:
        net_g = net_g_alex
        hps = hps_alex
    # elif voice == 4:
    #     net_g = net_g_nerea
    #     hps = hps_nerea
    # elif voice == 5:
    #     net_g = net_g_miren
    #     hps = hps_miren 
    # elif voice == 6:
    #     net_g = net_g_jon
    #     hps = hps_jon         
    else: 
        print("Voice not recognized. Using default (marina).") 
        net_g = net_g_marina 
        hps = hps_marina 
    
    net_g.to(device)
    stn_tst = get_text(question, hps, language='eu')
    with torch.no_grad(): 
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([sid]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy() 
        return audio
# def process_example(example, model, processor, kmeans):
#     try:
#         audio_path = os.path.join(
#             "/scratch/asudupe/datasets/VoiceAssistant-400K_eu",
#             example["answer_audio"]
#         )

#         waveform, sr = torchaudio.load(audio_path)
#         if waveform.ndim > 1:
#             waveform = waveform.squeeze(0)

#         tokens = assign_tokens(waveform, model, processor, kmeans)

#         # Compute global index from filename
#         global_idx = int(os.path.splitext(example["answer_audio"])[0].split("_")[-1])
#         filename = f"{global_idx:06d}.npy"

#         # Determine output subfolder based on range
#         range_dir = get_range_dir(global_idx)
#         out_dir = os.path.join(TOKEN_DIR, range_dir)
#         os.makedirs(out_dir, exist_ok=True)

#         out_path = os.path.join(out_dir, filename)
#         np.save(out_path, tokens)
#         example["answer_token"] = out_path
#     except:
#         example["answer_token"] = 'error'
#     return example

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

def process(ex, rank): 
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    # question = ex['question']
    answer = ex['answer'] 
    print(answer)

    # if len(question) > 450 or len(answer) > 1000:
    #     print("Skipping long example.")
    #     ex["question_audio"] = np.array([0.0], dtype=np.float32)
    #     ex["answer_audio"] = np.array([0.0], dtype=np.float32)
    #     return ex
    # random_voice = np.random.randint(0,11) 

    # print("Using voice:", random_voice)
    # spk = speakers[random_voice]
    # with open(output_file_path, 'a') as f:
    # try:
    #     audio_question = infer_voice(question, random_voice, device)

    #     audio_question = librosa.resample(audio_question, orig_sr=22050, target_sr=16000)
    #     ex["question_audio"] = audio_question
    #     ex['speaker'] = spk

    #     # logging.warning(f'Audio len: {len(audio)}')
    #     # audio_path_question = f'question_{idx}_{ex["index"]}_{spk}.wav'
    #     # sf.write(os.path.join("/scratch/asudupe/datasets/translation/audioak", audio_path_question), audio_question, 16000)
    #     # ex["audio_path_question"]=audio_path_question
    #     # f.write(audio_path+"\n")
    # except Exception as e:
    #     logging.warning(f"Error processing example: {e}")
    #     # audio_path = f'question_{idx}_{ex["index"]}_{spk}_error.wav'
    #     ex['question_audio'] = np.array([0.0], dtype=np.float32)
    #     ex['speaker'] = 'error'
    #     # ex["audio_path_question"]=audio_path

    try:
        audio_answer = infer_voice(answer, 9, device)
        audio_answer = librosa.resample(audio_answer, orig_sr=22050, target_sr=16000)
        # audio_path_answer = f'answer_{idx}_{ex["index"]}.wav'
        sf.write(os.path.join(ex["answer_audio"], audio_answer, 16000))
        tokens = assign_tokens(audio_answer, model, processor, kmeans)
        np.save(os.path.join(ex["answer_audio"],ex["answer_token"]), tokens)
        # ex["answer_audio"] = audio_answer
        # ex["audio_path_answer"]=audio_path_answer

    except:
        logging.warning(f"Error processing example: {ex['answer_audio']}")
        ex["answer_token"] = 'error'
        # audio_path = f'answer_{idx}_{ex["index"]}_error.wav'
        # ex["audio_path_answer"]=audio_path

        
    return ex

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    # parser.add_argument("--end", type=int)
    # parser.add_argument("--data_path", type=str)
    # parser.add_argument("--model_path", type=str)
    # multiprocess.set_start_method("spawn")
    
    print('Starting!')
    # args = parser.parse_args()
    multiprocess.set_start_method('forkserver', force=True)
    
    # output_path = os.path.join(args.data_path, f'audios_{args.start}_{args.end}/')
    # os.makedirs(output_path, exist_ok=True)

    # output_file_path=os.path.join(args.data_path, f'audios_{args.start}_{args.end}.txt')
    # if os.path.exists(output_file_path):
    #     with open(output_file_path, "rt") as f:
    #         progress = sum(1 for _ in f)
    # else:
    #     progress=0
    
    # file_path = os.path.join(
    #     args.data_path,
    #     "VoiceAssistant-400K"
    #     + f"_translated_{args.start}_{args.end}.jsonl",
    # )
    # # file_path = "/home/asudupe/Latxa-Omni/dataset_generation/translate_latxa/translation/clean.jsonl"
    # print(f'Opening the data in {file_path}')
    # data = []
    # with open(file_path, "r") as f:
    #     for i, line in enumerate(f):
    #         if i == 1:
    #             continue
    #         if line.strip():  # skip empty lines
    #             try:
    #                 data.append(json.loads(line))
    #             except:
    #                 print(line)
    #                 break

    # # Create Hugging Face dataset
    # dataset = Dataset.from_list(data)
    # def remove_notes(example):
    #     # Use regex to remove "Note:" and everything after
    #     example["answer"] = re.sub(r'(?i)\bnote\s*:\s*.*', '', example["answer"]).strip()
    #     return example

    # # Apply the cleaning function to the dataset
    # dataset = dataset.map(remove_notes)

    # dataset = dataset.select(range(progress, len(dataset), 1))

    # print(f'Starting the process! Dataset size: {len(dataset)}')
    



    # process_fn = partial(process, output_file_path=output_file_path)
    # print(process(dataset[0], 0))
    dataset = load_from_disk("/scratch/asudupe/datasets/VoiceAssistant-400K_eu/dataset_with_token_paths")
    n = len(dataset)

    parts = [
        dataset.select(range(0, n//4)),
        dataset.select(range(n//4, n//2)),
        dataset.select(range(n//2, 3*n//4)),
        dataset.select(range(3*n//4, n)),
    ]

    parts[args.id]

    dataset = dataset.map(process,
                        with_rank=True,
                        num_proc=4)

    print('Finished!')
    # dataset.save_to_disk(f"/scratch/asudupe/datasets/VoiceAssistant-400K_eu_{args.start}_{args.end}")

if __name__ == "__main__":
    main()