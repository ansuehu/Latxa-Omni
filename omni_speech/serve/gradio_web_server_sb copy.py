import argparse
import datetime
import json
import os
import time
import torch
import torchaudio

import gradio as gr
import numpy as np
import requests
import soundfile as sf

from omni_speech.conversation import default_conversation, conv_templates
from omni_speech.constants import LOGDIR
from omni_speech.utils import build_logger, server_error_msg
from speechbrain.inference.vocoders import UnitHIFIGAN

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
        # Added a timeout so it doesn't hang forever
        ret = requests.post(args.controller_url + "/refresh_all_workers", timeout=5)
        ret.raise_for_status() 
        ret = requests.post(args.controller_url + "/list_models", timeout=5)
        models = ret.json()["models"]
        logger.info(f"Models: {models}")
        return models
    except requests.exceptions.RequestException as e:
        logger.error(f"Could not connect to controller at {args.controller_url}: {e}")
        # Return a fallback list so the Gradio UI can still build the dropdown
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

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, None, "", "", None)


def add_speech(state, speech, request: gr.Request):
    text = "Please directly answer the questions in the user's speech."
    text = '<speech>\n' + text
    text = (text, speech)
    state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state)


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, chunk_size, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        yield (state, "", "", None)
        return

    if len(state.messages) == state.offset + 2:
        template_name = "llama_3"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]

    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, "", "", None)
        return

    prompt = state.get_prompt()
    sr, audio = state.messages[0][1][1]
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    audio = torch.tensor(audio.astype(np.float32)).unsqueeze(0)
    audio = resampler(audio).squeeze(0).numpy()
    audio /= 32768.0
    audio = audio.tolist()

    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1500),
        "stop": state.sep2,
        "audio": audio,
    }

    yield (state, "", "", None)

    output_unit = []
    output_text = ""
    
    # --- NEW: Variables to track streaming audio ---
    processed_unit_idx = 0 
    accumulated_audio = np.array([], dtype=np.float32)
    speaker_embedding = torch.tensor(np.load('/scratch/asudupe/models/hifigan/sonora_2/antton.npy'))

    # Initialize timing variables
    time_to_first_unit = None
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
                    output_text = data["text"][len(prompt):].strip()
                    output_unit = list(map(int, data["unit"].strip().split()))
                    state.messages[-1][-1] = (output_text, data["unit"].strip())

                    # --- NEW: Process audio chunks inside the loop ---
                    new_units = output_unit[processed_unit_idx:]
                    current_audio_yield = None if len(accumulated_audio) == 0 else (16000, accumulated_audio)

                    # If we have enough new units to meet the chunk_size threshold, vocode them
                    if len(new_units) >= int(chunk_size):
                        x = torch.LongTensor(new_units)
                        with torch.no_grad():
                            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
                        
                        wav_numpy = wav.detach().cpu().numpy()[0]
                        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))
                        processed_unit_idx += len(new_units)
                        
                        # Update the yield variable with the newly appended audio
                        current_audio_yield = (16000, accumulated_audio)

                    # Yield the progressively growing audio alongside the text
                    yield (state, output_text, data["unit"].strip(), current_audio_yield)
                else:
                    state.messages[-1][-1] = data["text"] + f" (error_code: {data['error_code']})"
                    yield (state, "", "", None)
                    return
                
                time.sleep(0.01) 
                
        stream_end = time.time()

    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, "", "", None)
        return

    # --- NEW: Process any remaining units after the stream ends ---
    remaining_units = output_unit[processed_unit_idx:]
    if len(remaining_units) > 0:
        x = torch.LongTensor(remaining_units)
        with torch.no_grad():
            wav = vocoder.decode_unit(x.unsqueeze(-1), speaker_embedding)
        wav_numpy = wav.detach().cpu().numpy()[0]
        accumulated_audio = np.concatenate((accumulated_audio, wav_numpy))
        
    final_audio = (16000, accumulated_audio) if len(accumulated_audio) > 0 else None
    audio_duration = len(accumulated_audio) / 16000.0 if len(accumulated_audio) > 0 else 0

    # Final yield
    yield (state, output_text, state.messages[-1][-1][1], final_audio)

    finish_tstamp = time.time()
    total_time = finish_tstamp - start_tstamp
    
    # Calculate Throughput and RTF safely
    streaming_time = (stream_end - stream_start) if stream_end else 0
    units_per_sec = len(output_unit) / streaming_time if streaming_time > 0 else 0
    rtf = total_time / audio_duration if audio_duration > 0 else 0

    # Log the detailed metrics
    logger.info(f"--- LLaMA-Omni Speed Metrics Chunk Size={chunk_size}---")
    logger.info(f"Time to First Unit (TTFU): {time_to_first_unit:.3f}s")
    logger.info(f"Streaming Throughput:      {units_per_sec:.2f} units/s")
    # logger.info(f"Vocoder Latency:           {vocoder_latency:.3f}s")
    logger.info(f"Audio Duration generated:  {audio_duration:.2f}s")
    logger.info(f"Total Response Time:       {total_time:.2f}s")
    logger.info(f"Real-Time Factor (RTF):    {rtf:.3f}x")

title_markdown = ("""
# 🎧 LLaMA-Omni: Seamless Speech Interaction with Large Language Models
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode, vocoder, cur_dir=None, concurrency_count=10):
    with gr.Blocks(title="LLaMA-Omni Speech Chatbot", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False)

        with gr.Row():
            audio_input_box = gr.Audio(sources=["upload", "microphone"], label="Speech Input")
            with gr.Accordion("Parameters", open=True) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature",)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max Output Tokens",)
                chunk_size = gr.Slider(minimum=10, maximum=500, value=40, step=10, interactive=True, label="Chunk Size",)

        if cur_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
        # gr.Examples(examples=[
        #     [f"{cur_dir}/examples/vicuna_1.wav"],
        #     [f"{cur_dir}/examples/vicuna_2.wav"],
        #     [f"{cur_dir}/examples/vicuna_3.wav"],
        #     [f"{cur_dir}/examples/vicuna_4.wav"],
        #     [f"{cur_dir}/examples/vicuna_5.wav"],
        #     [f"{cur_dir}/examples/helpful_base_1.wav"],
        #     [f"{cur_dir}/examples/helpful_base_2.wav"],
        #     [f"{cur_dir}/examples/helpful_base_3.wav"],
        #     [f"{cur_dir}/examples/helpful_base_4.wav"],
        #     [f"{cur_dir}/examples/helpful_base_5.wav"],
        # ], inputs=[audio_input_box])

        with gr.Row():
            submit_btn = gr.Button(value="Send", variant="primary")
            clear_btn = gr.Button(value="Clear")

        text_output_box = gr.Textbox(label="Text Output", type="text")
        unit_output_box = gr.Textbox(label="Unit Output", type="text") 
        audio_output_box = gr.Audio(label="Speech Output",autoplay=True)

        url_params = gr.JSON(visible=False)

        submit_btn.click(
            add_speech,
            [state, audio_input_box],
            [state]
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, chunk_size],
            [state, text_output_box, unit_output_box, audio_output_box],
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, audio_input_box, text_output_box, unit_output_box, audio_output_box],
            queue=False
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
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

    models = get_model_list()
    build_vocoder(args)

    logger.info(args)
    demo = build_demo(args.embed, vocoder, concurrency_count=args.concurrency_count)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )