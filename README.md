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

1. Launch a model worker.
   ```
   python -m omni_speech.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path Latxa-3.1-8B-Omni --s2s
   ```
