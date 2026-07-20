import argparse
import datetime
import json
import os
import time
import copy
import tempfile
import torch
import torchaudio

import gradio as gr
import numpy as np
import requests
import soundfile as sf

from omni_speech.constants import LOGDIR
from omni_speech.utils import build_logger, server_error_msg
from speechbrain.inference.vocoders import UnitHIFIGAN

os.environ['GRADIO_TEMP_DIR'] = './tmp'

logger = build_logger("gradio_web_server", "gradio_web_server.log")

vocoder = None

headers = {"User-Agent": "LLaMA-Omni Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    try:
        ret = requests.post(args.controller_url + "/refresh_all_workers", timeout=5)
        ret.raise_for_status() 
        ret = requests.post(args.controller_url + "/list_models", timeout=5)
        models = ret.json()["models"]
        logger.info(f"Models: {models}")
        return models
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to controller at {args.controller_url}: {e}")
        return ["Model controller offline"]


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    models = get_model_list()
    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    return [], dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return [], dropdown_update


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    # Returns empty state, clears input audio, clears output audio, clears chatbot
    return [], None, None, []


def http_bot(history_state, audio_input, model_selector, temperature, top_p, max_new_tokens, chunk_size, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]

    if worker_addr == "":
        history_state.append({"role": "assistant", "content": server_error_msg})
        yield (history_state, None, copy.deepcopy(history_state))
        return

    # # 1. Update the conversation history with the user's audio input path
    # if audio_input is not None:
    #     history_state.append({"role": "user", "content": {"path": audio_input}})
    
    # # 2. Process the Audio for the backend
    # # Load the audio file and resample to 16000Hz (the format Omni-1 expects)
    # waveform, sr = torchaudio.load(audio_input)
    # if waveform.shape[0] > 1: # Convert stereo to mono if necessary
    #     waveform = waveform.mean(dim=0, keepdim=True)
    # if sr != 16000:
    #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    #     waveform = resampler(waveform)
    
    # audio_list = waveform.squeeze(0).tolist()

    # # 3. Rebuild the Llama-3 prompt string dynamically from the history state
    # prompt = ""
    # for i, msg in enumerate(history_state):
    #     if msg["role"] == "user":
    #         # If this is the newest message, inject the <speech> tag to trigger audio extraction
    #         if i == len(history_state) - 1:
    #             prompt += "<|start_header_id|>user<|end_header_id|>\n\n<speech>\nPlease directly answer the questions in the user's speech.<|eot_id|>"
    #         # For past turns, remove the <speech> tag so the backend doesn't crash looking for old audio
    #         else:
    #             prompt += "<|start_header_id|>user<|end_header_id|>\n\n[User asked a previous question via audio]<|eot_id|>"
    #     elif msg["role"] == "assistant":
    #         # Extract just the text from previous assistant turns (ignore the audio filepath dicts)
    #         content = msg["content"]
    #         if isinstance(content, str):
    #             prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    
    # # Trigger the assistant to speak for the current turn
    # prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # # 4. Package the payload exactly as the Omni-1 backend expects it
    # pload = {
    #     "model": model_name,
    #     "prompt": prompt, 
    #     "audio": audio_list,
    #     "temperature": float(temperature),
    #     "top_p": float(top_p),
    #     "max_new_tokens": min(int(max_new_tokens), 1500),
    #     "stop": "<|eot_id|>",
    # }

    # 1. Update the conversation history with the user's audio input path
    if audio_input is not None:
        history_state.append({"role": "user", "content": {"path": audio_input}})
    
    # 2. Package the entire history in the payload (NO MORE prompt strings!)
    pload = {
        "model": model_name,
        "history": history_state, # <--- The worker will process this directly now!
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1500),
        "stop": "<|eot_id|>",
    }

    # Yield initial state to show user's message in the UI immediately
    yield (history_state, None, copy.deepcopy(history_state))

    output_unit = []
    processed_unit_idx = 0 
    accumulated_audio = np.array([], dtype=np.float32)
    speaker_embedding = torch.tensor(np.load('/scratch/asudupe/models/hifigan/sonora_2/antton.npy'))

    time_to_first_unit = None
    time_to_first_audio = None
    stream_start = time.time()
    stream_end = None

    try:
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if time_to_first_unit is None:
                time_to_first_unit = time.time() - start_tstamp

            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    
                    # Update text dynamically in the history state
                    if history_state[-1]["role"] == "assistant" and isinstance(history_state[-1]["content"], str):
                        history_state[-1]["content"] = data["text"]
                    else:
                        history_state.append({"role": "assistant", "content": data["text"]})

                    output_unit = list(map(int, data["unit"].strip().split()))
                    new_units = output_unit[processed_unit_idx:]

                    current_audio_yield = None

                    # If we have enough new units to meet the chunk_size threshold, vocode them
                    if len(new_units) >= int(chunk_size):
                        x = torch.LongTensor(new_units)
                        with torch.no_grad():
                            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
                        
                        wav_numpy = wav.detach().cpu().numpy()[0]
                        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))
                        processed_unit_idx += len(new_units)
                        
                        current_audio_yield = (16000, accumulated_audio)

                        if time_to_first_audio is None:
                            time_to_first_audio = time.time() - start_tstamp

                        # Yield the intermediate history state and progressive audio
                        yield (history_state, current_audio_yield, copy.deepcopy(history_state))
                    else:
                        # Yield just the text update if we don't have enough audio chunks yet
                        yield (history_state, None, copy.deepcopy(history_state))

                else:
                    error_output = data["text"] + f" (error_code: {data['error_code']})"
                    history_state.append({"role": "assistant", "content": error_output})
                    yield (history_state, None, copy.deepcopy(history_state))
                    return
                
                time.sleep(0.01) 
                
        stream_end = time.time()

    except requests.exceptions.RequestException as e:
        history_state.append({"role": "assistant", "content": server_error_msg})
        yield (history_state, None, copy.deepcopy(history_state))
        return

    # Process any remaining audio units left over after the stream finishes
    remaining_units = output_unit[processed_unit_idx:]
    if len(remaining_units) > 3:
        x = torch.LongTensor(remaining_units)
        with torch.no_grad():
            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
        wav_numpy = wav.detach().cpu().numpy()[0]
        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))
        
    audio_duration = len(accumulated_audio) / 16000.0 if len(accumulated_audio) > 0 else 0

    # 3. Finalize: Save the completely accumulated audio to a temporary file and append to history
    if len(accumulated_audio) > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, accumulated_audio, 16000)
            # history_state.append({"role": "assistant", "content": {"path": f.name, "type": "audio/wav"}})
    
    # Final yield to fully update the Chatbot UI with the generated audio file
    yield (history_state, None, copy.deepcopy(history_state))

    finish_tstamp = time.time()
    total_time = finish_tstamp - start_tstamp
    
    # Calculate Throughput and RTF safely
    streaming_time = (stream_end - stream_start) if stream_end else 0
    units_per_sec = len(output_unit) / streaming_time if streaming_time > 0 else 0
    rtf = total_time / audio_duration if audio_duration > 0 else 0

    # Log the detailed metrics
    # logger.info(f"--- LLaMA-Omni Speed Metrics Chunk Size={chunk_size}---")
    # logger.info(f"Time to First Unit (TTFU): {time_to_first_unit if time_to_first_unit else 0:.3f}s")
    # if time_to_first_audio:
    #     logger.info(f"Time to First Audio (TTFA):{time_to_first_audio:.3f}s") 
    # logger.info(f"Streaming Throughput:      {units_per_sec:.2f} units/s")
    # logger.info(f"Audio Duration generated:  {audio_duration:.2f}s")
    # logger.info(f"Total Response Time:       {total_time:.2f}s")
    # logger.info(f"Real-Time Factor (RTF):    {rtf:.3f}x")
    print(history_state)


title_markdown = ("""
# 🎧 LLaMA-Omni: Seamless Speech Interaction with Large Language Models
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

def build_demo(embed_mode, vocoder, cur_dir=None, concurrency_count=10):
    models = get_model_list()

    with gr.Blocks(title="LLaMA-Omni Speech Chatbot", theme=gr.themes.Default(), css=block_css) as demo:
        # Replaced custom conversation class with standard list state
        history_state = gr.State([])

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False)
        
        # ADDED: Integrated the Chatbot UI component
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            type="messages",
            scale=1,
        )

        with gr.Row():
            audio_input_box = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speech Input")
            
            # This handles the intermediate stream playback without showing download buttons for incomplete streams
            audio_output_box = gr.Audio(label="Speech Output", show_download_button=False, autoplay=False, visible=False)

        with gr.Accordion("Parameters", open=True) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature",)
            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max Output Tokens",)
            chunk_size = gr.Slider(minimum=10, maximum=500, value=40, step=10, interactive=True, label="Chunk Size",)

        with gr.Row():
            submit_btn = gr.Button(value="Send", variant="primary")
            clear_btn = gr.Button(value="Clear")

        url_params = gr.JSON(visible=False)

        submit_btn.click(
            http_bot,
            inputs=[
                history_state, 
                audio_input_box, 
                model_selector, 
                temperature, 
                top_p, 
                max_output_tokens, 
                chunk_size
            ],
            outputs=[history_state, audio_output_box, chatbot],
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [history_state, audio_input_box, audio_output_box, chatbot],
            queue=False
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [history_state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [history_state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


def build_vocoder(args):
    global vocoder
    if args.vocoder is None:
        return None
    vocoder = UnitHIFIGAN.from_hparams(source=args.vocoder, run_opts={"device":'cuda'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--vocoder", type=str)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    build_vocoder(args)

    logger.info(args)
    demo = build_demo(args.embed, vocoder, concurrency_count=args.concurrency_count)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )