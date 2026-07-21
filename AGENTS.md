# Latxa-Omni

Basque speech-to-speech (S2S) system built on Llama-Omni, using
Latxa-3.1-8B-Instruct as the LLM backbone.

## Architecture

Three-model serving stack: **Controller → Model Worker(s) → Gradio/FastAPI**.

- Speech encoder: Whisper (large-v3, `models/speech_encoder/large-v3.pt`)
- Speech projector: Linear (`config-Llama-3.1-8B-Instruct.json`)
- LLM backbone: Latxa-3.1-8B-Instruct (8K→131K context via RoPE scaling x8)
- Speech generator: CTC, upsample factor 25, vocab size 1000
- Vocoder: HiFi-GAN (UnitHIFIGAN, Basque speaker embeddings)
- Conversation mode: `llama_3`, token `<speech>` at index -200
- Input: whisper log-mel spectrogram (128 bands)

Two-stage training: Stage 1 (speech projector pretraining, LM frozen) → Stage 2 (full S2S with CTC loss + tgt_units).

## No package install

No `setup.py`/`pyproject.toml`. `export PYTHONPATH=$(pwd)` required. The
`latxa` shell command (alias to load environment/CUDA) must run before any
command.

## Key commands

```bash
# ---------- Serving ----------
# 1. controller (background)
python -m omni_speech.serve.controller --host 0.0.0.0 --port 10000

# 2. model worker
python -m omni_speech.serve.model_worker --host 0.0.0.0 \
  --controller http://localhost:10000 --port 40000 \
  --worker http://localhost:40000 --model-path Latxa-3.1-8B-Omni --s2s

# 3. web UI (pick one variant)
python -m omni_speech.serve.gradio_web_server_history \
  --controller http://localhost:10000 --port 8000 \
  --model-list-mode reload --vocoder ./HiFiGAN-Basque-Maider-Antton

# FastAPI backend
uvicorn omni_speech.web.main:app --host 0.0.0.0 --port 8000

# ---------- Training ----------
# Stage 1 (projector pretrain)
python omni_speech/train/stage1.py \
  --model-path <base_llm> --train-file <json/arrow> ...

# Stage 2 (full S2S)
python omni_speech/train/stage2.py \
  --model-path <stage1_ckpt> --model-base <base_llm> \
  --train-file <hf_dataset> ...

# Multi-GPU via torchrun
torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29400 \
  omni_speech/train/stage1.py ...

# ---------- Inference ----------
python omni_speech/infer/infer.py \
  --model-path Latxa-3.1-8B-Omni --model-base HiTZ/Latxa-Llama-3.1-8B-Instruct \
  --question-file <json> --answer-file <jsonl> --s2s

# ---------- Evaluation ----------
python omni_speech/evaluate/evaluate_wer.py
python omni_speech/evaluate/calculate_utmos.py
python omni_speech/evaluate/evaluate_chatgpt_score.py
```

## Quirks & gotchas

- `stage1.py` has hardcoded `os.chdir("/home/asudupe/Latxa-Omni")` at line 3.
- SLURM scripts hardcode `/scratch/asudupe/` checkpoint paths, require SSL
  cert at `~/cacert.pem`, and use NCCL env vars (`CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `TORCH_NCCL_BLOCKING_WAIT=1`).
- Gradio has 3 variants: `gradio_web_server_history.py`,
  `gradio_web_server_history_new.py`, `gradio_web_server_sb.py`.
- Model loading uses `load_pretrained_model` (inference) or `create_model`
  (training); pick `s2s=True` for `OmniSpeech2SLlamaForCausalLM` else
  `OmniSpeechLlamaForCausalLM` (text-only).
- `create_model` in training freezes speech encoder (`requires_grad = False`).
- WANDB logging via env vars: `WANDB_PROJECT`, `WANDB_LOG_MODEL`, `WANDB_WATCH`.
- `pad_token_id=128004` (`<|finetune_right_pad_id|>`) used in generation.
- Tokenizer: `use_fast=False` for `load_pretrained_model`, `use_fast=True` for
  `create_model`.
- CTC postprocess: deduplicate then filter blank (= `unit_vocab_size`, i.e. 1000).

## Git-ignored paths (restore after clone)

`saves/`, `models/`, `audioak/`, `results/`, `ebaluazioa/`, `vocoder/`,
`wandb/`, `*.log`, `HiFiGAN-Basque-Maider-Antton/`, `Latxa-3.1-8B-Omni/`.

## Repository layout

```
omni_speech/           # Main package
├── model/             # speech_encoder, speech_projector, speech_generator, language_model
├── train/             # stage1.py, stage2.py, SLURM/shell scripts, DeepSpeed configs
├── serve/             # controller, model_worker, gradio_web_server variants
├── datasets/          # preprocess.py
├── infer/             # batch inference
├── evaluate/          # WER, UTMOS, ChatGPT score
└── web/               # FastAPI backend
models/                # speech encoder checkpoints (large-v3.pt, mHubert, NeMo)
configs/               # model config JSONs
dataset_generation/    # HuBERT token dump, VITS, translation
```

## External model checkpoints

- `Latxa-3.1-8B-Omni` → `ansuehu/Latxa-3.1-8B-Omni` on HuggingFace
- `HiFiGAN-Basque-Maider-Antton` → `Ansu/HiFiGAN-Basque-Maider-Antton`
- Speech encoder: `models/speech_encoder/large-v3.pt` (Whisper)
