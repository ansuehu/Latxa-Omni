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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from speechbrain.inference.vocoders import UnitHIFIGAN
import subprocess
import uuid
import shutil

app = FastAPI(title="Latxa-Omni API")
conversations = {}

# Configuration
CONTROLLER_URL = "http://localhost:10000"
MODEL_NAME = "None"
VOCODER_PATH = "/dipc/asudupe/Latxa-Omni/HiFiGAN-Basque-Maider-Antton"
SPEAKER_EMBEDDING_PATH = '/scratch/asudupe/models/hifigan/sonora_2/antton.npy'
AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "audio_output")

os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# Load vocoder globally
vocoder = None
speaker_embedding = None

headers = {"User-Agent": "LLaMA-Omni Client"}


def to_backend_history(history_state):
    """Convert UI history to the format the model worker expects.

    Worker expects assistant turns as plain text strings (not dicts with paths).
    User audio paths stored as filenames are resolved to absolute paths.
    """
    backend = []
    for msg in history_state:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            path = content.get("path", "") if isinstance(content, dict) else ""
            if path and not os.path.isabs(path):
                path = os.path.join(AUDIO_OUTPUT_DIR, path)
            backend.append({"role": "user", "content": {"path": path}})
        else:
            if isinstance(content, str):
                backend.append({"role": "assistant", "content": content})
            elif isinstance(content, dict):
                backend.append({"role": "assistant", "content": content.get("text", "")})
    return backend


@app.on_event("startup")
def load_models():
    global vocoder, speaker_embedding
    print("Loading vocoder...")
    vocoder = UnitHIFIGAN.from_hparams(source=VOCODER_PATH, run_opts={"device": 'cuda'})
    speaker_embedding = torch.tensor(np.load(SPEAKER_EMBEDDING_PATH))
    print("Models loaded.")


# Mount the static directory for the HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    hist = conversations.get(session_id, [])
    return JSONResponse({"session_id": session_id, "history": hist})


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(filepath, media_type="audio/wav")


@app.post("/api/chat")
async def chat_endpoint(audio: UploadFile = File(...), session_id: str = Form("default_session")):
    # 1. Save incoming WebM audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        webm_path = temp_audio.name

    wav_path = webm_path.replace(".webm", ".wav")

    # 2. Convert WebM to 16kHz Mono WAV using FFmpeg
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", webm_path,
            "-ac", "1", "-ar", "16000", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        os.unlink(webm_path)
        raise HTTPException(status_code=500, detail="FFmpeg conversion failed.")
    except FileNotFoundError:
        os.unlink(webm_path)
        raise HTTPException(status_code=500, detail="FFmpeg is not installed.")

    os.unlink(webm_path)

    if session_id not in conversations:
        conversations[session_id] = []

    history_state = conversations[session_id]

    # 3. Copy user audio to output dir with a stable filename for serving
    user_audio_filename = f"user_{session_id}_{uuid.uuid4().hex}.wav"
    user_audio_path = os.path.join(AUDIO_OUTPUT_DIR, user_audio_filename)
    shutil.copy2(wav_path, user_audio_path)
    os.unlink(wav_path)

    history_state.append({"role": "user", "content": {"path": user_audio_filename}})

    # 4. Dynamically fetch the model and worker address
    try:
        requests.post(f"{CONTROLLER_URL}/refresh_all_workers", timeout=5)
        ret_models = requests.post(f"{CONTROLLER_URL}/list_models", timeout=5)
        ret_models.raise_for_status()
        models = ret_models.json().get("models", [])

        if not models:
            raise HTTPException(status_code=503, detail="No models available.")

        active_model = models[0]
        print(f"Active model found: {active_model}")

        ret_addr = requests.post(f"{CONTROLLER_URL}/get_worker_address", json={"model": active_model}, timeout=5)
        ret_addr.raise_for_status()
        worker_addr = ret_addr.json().get("address", "")

        if not worker_addr:
            raise HTTPException(status_code=503, detail="Worker offline.")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Controller error: {str(e)}")

    # 5. Build the payload using a backend-friendly view of the history.
    pload = {
        "model": active_model,
        "history": to_backend_history(history_state),
        "temperature": 0.0,
        "top_p": 0.7,
        "max_new_tokens": 512,
        "stop": "<|eot_id|>",
    }

    # 6. Append an assistant placeholder that will be filled as the stream progresses
    history_state.append({"role": "assistant", "content": {"text": "", "path": None}})

    output_unit = []
    processed_unit_idx = 0
    accumulated_audio = np.array([], dtype=np.float32)
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
                    assistant_text_response = data.get("text", "")
                    history_state[-1]["content"]["text"] = assistant_text_response

                    output_unit = list(map(int, data["unit"].strip().split()))
                    new_units = output_unit[processed_unit_idx:]

                    if len(new_units) >= chunk_size:
                        x = torch.LongTensor(new_units)
                        with torch.no_grad():
                            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
                        wav_numpy = wav.detach().cpu().numpy()[0]
                        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))
                        processed_unit_idx += len(new_units)

                else:
                    error_output = data["text"] + f" (error_code: {data['error_code']})"
                    history_state[-1]["content"]["text"] = error_output
                    conversations[session_id] = history_state
                    return JSONResponse({
                        "error": error_output,
                        "text": error_output,
                        "history": history_state
                    })

                time.sleep(0.01)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Worker error: {str(e)}")

    # 7. Process remaining units
    remaining_units = output_unit[processed_unit_idx:]
    if len(remaining_units) > 3:
        x = torch.LongTensor(remaining_units)
        with torch.no_grad():
            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
        wav_numpy = wav.detach().cpu().numpy()[0]
        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))

    # 8. Save the generated audio and attach to history
    audio_filename = f"{session_id}_{uuid.uuid4().hex}.wav"
    audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)

    if len(accumulated_audio) > 0:
        sf.write(audio_path, accumulated_audio, 16000)
        history_state[-1]["content"]["path"] = audio_filename

    conversations[session_id] = history_state

    return JSONResponse({
        "text": assistant_text_response,
        "audio_url": f"/audio/{audio_filename}" if len(accumulated_audio) > 0 else None,
        "history": history_state
    })
