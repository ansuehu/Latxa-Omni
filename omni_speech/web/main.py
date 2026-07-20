import os
import time
import json
import tempfile
import torch
import torchaudio
import numpy as np
import requests
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from speechbrain.inference.vocoders import UnitHIFIGAN
import subprocess

app = FastAPI(title="Latxa-Omni API")
conversations = {}

# Configuration
CONTROLLER_URL = "http://localhost:10000"
MODEL_NAME = "None" # Update this to your specific model name
VOCODER_PATH = "/dipc/asudupe/Latxa-Omni/HiFiGAN-Basque-Maider-Antton" # Update to your vocoder path
SPEAKER_EMBEDDING_PATH = '/scratch/asudupe/models/hifigan/sonora_2/antton.npy'

# Load vocoder globally
vocoder = None
speaker_embedding = None

@app.on_event("startup")
def load_models():
    global vocoder, speaker_embedding
    print("Loading vocoder...")
    vocoder = UnitHIFIGAN.from_hparams(source=VOCODER_PATH, run_opts={"device":'cuda'})
    speaker_embedding = torch.tensor(np.load(SPEAKER_EMBEDDING_PATH))
    print("Models loaded.")

# Mount the static directory for the HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/api/chat")
async def chat_endpoint(audio: UploadFile = File(...), session_id: str = Form("default_session")):
    # 1. Save incoming WebM audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        webm_path = temp_audio.name

    wav_path = webm_path.replace(".webm", ".wav")

    # 2. Convert WebM to 16kHz Mono WAV using FFmpeg
    try:
        # This command forces 1 channel (-ac 1) and 16000Hz sample rate (-ar 16000)
        subprocess.run([
            "ffmpeg", "-y", "-i", webm_path,
            "-ac", "1", "-ar", "16000", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except subprocess.CalledProcessError as e:
        print(f"🚨 FFMPEG CONVERSION ERROR: {e}")
        raise HTTPException(status_code=500, detail="FFmpeg conversion failed.")
    except FileNotFoundError:
        print("🚨 FFMPEG NOT FOUND: Please install ffmpeg on your server.")
        raise HTTPException(status_code=500, detail="FFmpeg is not installed.")
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    history_state = conversations[session_id]

    # Update the conversation history with the user's audio input path
    history_state.append({"role": "user", "content": {"path": wav_path}})
    
    # 3. Dynamically fetch the model and worker address
    try:
        # Refresh the worker list first
        requests.post(f"{CONTROLLER_URL}/refresh_all_workers", timeout=5)
        
        # Fetch available models
        ret_models = requests.post(f"{CONTROLLER_URL}/list_models", timeout=5)
        ret_models.raise_for_status()
        models = ret_models.json().get("models", [])
        
        if not models:
            print("🚨 CONTROLLER ERROR: No models are currently registered with the controller.")
            raise HTTPException(status_code=503, detail="No models available.")
            
        # Select the first available model dynamically
        active_model = models[0]
        print(f"✅ Active model found: {active_model}")
        
        # Now get the worker address for this specific model
        ret_addr = requests.post(f"{CONTROLLER_URL}/get_worker_address", json={"model": active_model}, timeout=5)
        ret_addr.raise_for_status()
        worker_addr = ret_addr.json().get("address", "")
        
        if not worker_addr:
            print(f"🚨 CONTROLLER ERROR: No address returned for model '{active_model}'.")
            raise HTTPException(status_code=503, detail="Worker offline.")
            
    except requests.exceptions.RequestException as e:
        print(f"🚨 CONTROLLER COMMUNICATION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Controller error: {str(e)}")

    # 4. Build the payload using the dynamically fetched active_model
    pload = {
        "model": active_model,
        "history": history_state,
        "temperature": 0.0,
        "top_p": 0.7,
        "max_new_tokens": 512,
        "stop": "<|eot_id|>",
    }

    # 4. Stream response and vocode units
    headers = {"User-Agent": "LLaMA-Omni Client"}
    accumulated_audio = np.array([], dtype=np.float32)
    output_unit = []
    processed_unit_idx = 0
    chunk_size = 40
    assistant_text_response = ""

    try:
        response = requests.post(
            f"{worker_addr}/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10
        )
        
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    # Capture the text dynamically as it streams in
                    assistant_text_response = data.get("text", "")

                    output_unit = list(map(int, data["unit"].strip().split()))
                    new_units = output_unit[processed_unit_idx:]

                    # Vocode chunks
                    if len(new_units) >= chunk_size:
                        x = torch.LongTensor(new_units)
                        with torch.no_grad():
                            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
                        wav_numpy = wav.detach().cpu().numpy()[0]
                        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))
                        processed_unit_idx += len(new_units)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Worker error: {str(e)}")

    # Process remaining units
    remaining_units = output_unit[processed_unit_idx:]
    if len(remaining_units) > 3:
        x = torch.LongTensor(remaining_units)
        with torch.no_grad():
            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
        wav_numpy = wav.detach().cpu().numpy()[0]
        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))

    # 6. Save and append the generated audio
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(out_file.name, accumulated_audio, 16000)

    # ✅ ONLY append the text context. 
    print(assistant_text_response)
    history_state.append({"role": "assistant", "content": assistant_text_response})
    
    # 🚨 DELETE THIS LINE entirely so it doesn't pollute the LLM's context:
    # history_state.append({"role": "assistant", "content": {"path": out_file.name, "type": "audio/wav"}})

    print(history_state)

    # Ensure dictionary maintains the updated list reference
    conversations[session_id] = history_state

    return FileResponse(
        out_file.name, 
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"}
    )