# import matplotlib.pyplot as plt
# import IPython.display as ipd

import os
# import json
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

import soundfile
import speech
import numpy as np
import time
# import sys
from datasets import load_dataset
import multiprocess
import re



hps_marina = get_hparams_from_file("/scratch/asudupe/models/vits/configs/sonora.json")

net_g_marina = SynthesizerTrn(
    len(symbols),
    hps_marina.data.filter_length // 2 + 1,
    hps_marina.train.segment_size // hps_marina.data.hop_length,
    **hps_marina.model).cuda()
_ = net_g_marina.eval()

_ = load_checkpoint("./checkpoints/marina_898.pth", net_g_marina, None)

hps_aintzane = get_hparams_from_file("/scratch/asudupe/models/vits/configs/aintzane_eu_3.json")

net_g_aintzane = SynthesizerTrn(
    len(symbols),
    hps_aintzane.data.filter_length // 2 + 1,
    hps_aintzane.train.segment_size // hps_aintzane.data.hop_length,
    **hps_aintzane.model).cuda()
_ = net_g_aintzane.eval()

_ = load_checkpoint("/scratch/asudupe/models/vits/checkpoints/aintzane_3382.pth", net_g_aintzane, None)

hps_kiko = get_hparams_from_file("/scratch/asudupe/models/vits/configs/kiko_eu_2.json")

net_g_kiko = SynthesizerTrn(
    len(symbols),
    hps_kiko.data.filter_length // 2 + 1,
    hps_kiko.train.segment_size // hps_kiko.data.hop_length,
    **hps_kiko.model).cuda()
_ = net_g_kiko.eval()

_ = load_checkpoint("/scratch/asudupe/models/vits/checkpoints/kiko_4374.pth", net_g_kiko, None)

hps_alex = get_hparams_from_file("/scratch/asudupe/models/vits/configs/sonora.json")

net_g_alex = SynthesizerTrn(
    len(symbols),
    hps_alex.data.filter_length // 2 + 1,
    hps_alex.train.segment_size // hps_alex.data.hop_length,
    **hps_alex.model).cuda()
_ = net_g_alex.eval()

_ = load_checkpoint("/scratch/asudupe/models/vits/checkpoints/alex_864.pth", net_g_alex, None)

hps_kristof = get_hparams_from_file("./configs/kristof_eu.json")

net_g_kristof = SynthesizerTrn(
    len(symbols),
    hps_kristof.data.filter_length // 2 + 1,
    hps_kristof.train.segment_size // hps_kristof.data.hop_length,
    **hps_kristof.model).cuda()
_ = net_g_kristof.eval()

_ = load_checkpoint("./checkpoints/kristof_1697.pth", net_g_kristof, None)

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
    clp = ""
    for p in range(len(phones)):
        clp = clp + "".join(phones[p].split('-'))
    #     # print(f"clp: {clp}, phone: {phones[p]}")
    #     if p == len(phones) - 1:
    #         clp = clp + ' | '
    
    slp = str(clp).split()
    # print(slp)
    if '?' in checker:
        slp.append('?')
    elif '!' in checker:
        slp.append('!')
    elif '.' in checker:
        slp.append('.')
    else:
        slp.append('.')
    
    if checker[-2] == ':' or checker[-2] == ';':
        slp.append('.')
    phones = np.array(slp)

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
    if voice == 0:
        net_g = net_g_marina
        hps = hps_marina
    elif voice == 1:
        net_g = net_g_kiko
        hps = hps_kiko 
    elif voice == 2:
        net_g = net_g_alex
        hps = hps_alex
    elif voice == 3:
        net_g = net_g_kristof
        hps = hps_kristof
    elif voice == 4:
        net_g = net_g_aintzane
        hps = hps_aintzane 
    else: 
        print("Voice not recognized. Using default (marina).") 
        net_g = net_g_marina 
        hps = hps_marina 
    
    net_g.to(device)
    stn_tst = get_text(question, hps, language='eu')
    with torch.no_grad(): 
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy() 
        return audio 
    
def process(ex, rank): 
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    question = ex['question'][8:] 
    answer = ex['answer'] 

    if len(question) > 450 or len(answer) > 1000:
        print("Skipping long example.")
        ex["question_audio"] = np.array([0.0], dtype=np.float32)
        ex["answer_audio"] = np.array([0.0], dtype=np.float32)
        return ex
    random_voice = np.random.randint(0,5) 
    # print("Using voice:", random_voice)
    try:
        ex["question_audio"] = infer_voice(question, random_voice, device)
        ex["answer_audio"] = infer_voice(answer, 4, device)
    except Exception as e:
        print("Error processing example:", e)
        ex["question_audio"] = np.array([0.0], dtype=np.float32)
        ex["answer_audio"] = np.array([0.0], dtype=np.float32)
    return ex

def main(): 
    # multiprocess.set_start_method("spawn")
    
    multiprocess.set_start_method('forkserver', force=True)
    dataset = load_dataset("Ansu/Instruct_200k_eu_filtered_8", split="train")

    # print(process(dataset[0], 0))
    dataset = dataset.map(process,
                        with_rank=True,
                        num_proc=torch.cuda.device_count() * 4)

    dataset.push_to_hub("Ansu/Instruct_S2S_eu", private=False, token='...')

if __name__ == "__main__":
    main()
