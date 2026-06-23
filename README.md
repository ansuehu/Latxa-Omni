# Latxa-Omni
Reproducing [Llama-Omni](https://github.com/ictnlp/LLaMA-Omni) with Latxa-3.1-8B-Instruct for basque S2S. This repository adapts the training code of Llama-Omni reproduced by [wntg](https://github.com/wntg/LLaMA-Omni) and changed by [chiawen](https://github.com/chiawen0104/llama-omni-ckip_pa).

## Create Conda Environment
1. Clone the repository.
   ```
   git clone https://github.com/ictnlp/LLaMA-Omni
   cd LLaMA-Omni
   ```
2. Ensure CUDA version 12.1 is loaded
   ```
   module load CUDA/12.1.1
   ```
4. Install packages.
   ```
   conda create -n llama-omni python=3.10
   conda activate llama-omni
   pip install pip==24.0
   pip install -e .
   ```
5. Change some packages versions:
   ```
   pip install transformers==4.45.0 deepspeed==0.15.4 accelerate==0.34.2 pydantic==2.8.2 wandb datasets
   conda install ffmpeg
   ```
6. Install `flash-attention` (v2) for the right CUDA and Torch versions.
   ```
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ```
   If the installation fails, please visit [here](https://github.com/Dao-AILab/flash-attention/releases) to see the wheel files, and then rerun the above command.

## Installation
1. Clone this repository.
   ```
   git https://github.com/ansuehu/Latxa-Omni
   cd Latxa-Omni
   ```
2. Download the `Llama-3.1-8B-Omni` model from [Huggingface](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni).
   ```
   pip install huggingface_hub
   huggingface-cli login
   ```
   ```
   huggingface-cli download ansuehu/Latxa-3.1-8B-Omni --local-dir ./Latxa-3.1-8B-Omni
   huggingface-cli download Ansu/HiFiGAN-Basque-Maider-Antton --local-dir ./HiFiGAN-Basque-Maider-Antton
   ```
## Gradio Demo
1. Launch a controller.
   ```
   python -m omni_speech.serve.controller --host 0.0.0.0 --port 10000
   ```

2. Launch a gradio web server.
   ```
   python -m omni_speech.serve.gradio_web_server_sb --controller http://localhost:10000 --port 8000 --model-list-mode reload --vocoder ./HiFiGAN-Basque-Maider-Antton 
   ```

3. Launch a model worker.
   ```
   python -m omni_speech.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path Latxa-3.1-8B-Omni --s2s
   ```

For port forwarding with ProxyJump:

ssh -J jumpuser@jumphost targetuser@targethost -L localport:targethost:targetport
## Local Inference
   ```
   import torch
   import torchaudio
   from omni_speech.conversation import conv_templates, SeparatorStyle
   from omni_speech.model.builder import load_pretrained_model
   from omni_speech.datasets.preprocess import tokenizer_speech_token
   import whisper
   import numpy as np
   from speechbrain.inference.vocoders import UnitHIFIGAN

   def ctc_postprocess(tokens, blank):
       _toks = tokens.squeeze(0).tolist()
       deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
       hyp = [v for v in deduplicated_toks if v != blank] #官方493 222
       hyp = " ".join(list(map(str, hyp))) #1918 547
       return hyp

   model_path = "Latxa-3.1-8B-Omni"
   model_base = None
   is_lora = False
   s2s = True
   mel_size = 128
   conv_mode = 'llama_3'
   tokenizer, model, context_len = load_pretrained_model(model_path, model_base, is_lora=is_lora, s2s=s2s)

   hifigan = UnitHIFIGAN.from_hparams(source="HiFiGAN-Basque-Maider-Antton", run_opts={"device":'cuda'})

   qs = "<speech>\nPlease answer the questions in the user's speech with a few words."
   speech_file = 'path/to/audio'
   speech_loaded = whisper.load_audio(speech_file)
   
   conv = conv_templates[conv_mode].copy()
   conv.append_message(conv.roles[0], qs)
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   
   speech = whisper.pad_or_trim(speech_loaded)
   speech = whisper.log_mel_spectrogram(speech, n_mels=mel_size).permute(1, 0)
   
   input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors='pt')
   speech_length = torch.LongTensor([speech.shape[0]])
   
   input_ids = input_ids.to(device='cuda', non_blocking=True)
   speech_tensor = speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
   speech_length = speech_length.to(device='cuda', non_blocking=True)
   
   input_ids = input_ids.unsqueeze(0)
   speech_tensors = speech_tensor.unsqueeze(0)
   speech_lengths = speech_length.unsqueeze(0)
   
   temperature = 0
   top_p = None   
   num_beams = 1
   max_new_tokens = 512
   
   with torch.inference_mode():
       outputs = model.generate(
           input_ids,
           speech=speech_tensors,
           speech_lengths=speech_lengths,
           do_sample=True if temperature > 0 else False,
           temperature=temperature,
           top_p=top_p,
           num_beams=num_beams,
           max_new_tokens=max_new_tokens,
           use_cache=True,
           pad_token_id=128004,
           streaming_unit_gen=True,
    
       )
   output_ids, output_units = outputs
   
   print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
   output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)
   output_units = torch.tensor([int(x) for x in output_units.split()], dtype=torch.long)
   answer = hifigan.decode_unit(output_uni·ts.unsqueeze(-1), torch.tensor(np.load('HiFiGAN-Basque-Maider-Antton/speaker_embeddings/antton.npy')))
   torchaudio.save("erantzuna.wav", answer.cpu(), sample_rate=16000)
   ```
