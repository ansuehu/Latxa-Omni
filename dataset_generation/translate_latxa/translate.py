from argparse import ArgumentParser
import json
import os
from typing import List, Literal
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel, RootModel
import logging
from datasets import load_from_disk, Audio
import time
import re
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
class Conversation(RootModel):
    root: List[Message]

examples =[
    {"role": "system",
     "content": """You are a helpful AI assistant that specializes in English to Basque translations.
    Your task is to translate instruction datasets from English to Basque.
    Here are some important guidelines:
    1. Maintain the original meaning and intent of the instructions
    2. If the original sentence is about something cultural change it to a similar Basque culture reference.
    3. If the conversation is language-dependent (for example, relies on English idioms, jokes, wordplay, or culture-specific expressions) change it a similar Basque sentence.
    4. If the conversation refers to some context not present invent something.
    5. Use standard Basque language (batua)
    6. Keep the technical terms that don't have widely accepted Basque translations
    7. In the original sentences it refers to himself as Omni, use Latxa-Omni instead.
    
    Please provide accurate Basque translations for all text fields.
    """
    },
    {
        "role": "user",
        "content": f"Translate the following multiple choice questions and answers to Basque \n\n\n Question: Can you help me with my homework? \n Answer: Sure, what subject is it?"
    },
    {
        "role": "assistant",
        "content": f"Galdera: Lagundu ahal didazu nire etxeko lanekin? \n Erantzuna: Bai, zein ikasgai da?"
    },
    {
        "role": "user",
        "content": f"Translate the following question and answers to Basque \n\n\n Question: What is the most emblematic thing in London? \n Answer: The most emblematic thing in London is Big Ben."
    },
    {
        "role": "assistant",
        "content": f"Galdera: Zer da Bilboko gauzarik enblematikoena? \n Erantzuna: Bilboko gauzarik enblematikoena Guggenheim Museoa da."
    }
]
    
def load_dataset_slice(dataset_path, slice):
    dataset = load_from_disk(dataset_path)
    dataset = dataset["train"].cast_column("question_audio", Audio(decode=False))
    if slice[-1]>=len(dataset):
        slice = range(slice[0], len(dataset), 1)
    return dataset.select(slice)

def batch_generator(dataset, batch_size=1):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def prepare_example(question, answer):
    prompt = [{"role": "user", "content": f"Translate the following question and answers to Basque \n\n\n Question: {question} \n Answer: {answer}"}]
    return examples + prompt

def postprocess_output(output):
    instruction = output[(output.index("Galdera:") + len("Galdera:")):(output.index("Erantzuna:") -3)]
    response = output[output.index("Erantzuna:") + len("Erantzuna:"):]
    #Remove leading spaces
    instruction = instruction.strip()
    response = response.strip()
    return instruction, response

def main(args):
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        download_dir="/scratch/emiranda/cache_00/",
        max_model_len=25700,
        guided_decoding_backend="outlines"
    )
    galdera_rule = r'Galdera:[\s\S]*Erantzuna:[\s\S]*'
    compiled_rule = re.compile(galdera_rule) # Compile the regex
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        stop="\n'''",
        guided_decoding=GuidedDecodingParams(
            regex=galdera_rule,
            backend="outlines"  # Uncomment if you want to force a specific backend
        ),
        frequency_penalty=args.frequency_penalty,
    )
    output_file_path = os.path.join(
        args.output_path,
        os.path.basename(args.dataset_path)
        + f"_translated_{args.dataset_start}_{args.dataset_end}.jsonl",
    )
    if os.path.exists(output_file_path):
        with open(output_file_path, "rt") as f:
            progress = sum(1 for _ in f)
    else:
        progress = 0
    logging.info('Loading dataset')
    hasi = time.time()
    dataset = load_dataset_slice(
        args.dataset_path, range(args.dataset_start + progress, args.dataset_end, 1)
    )
    logging.info(f'Dataset loaded, took {time.time()-hasi}s')
    logging.info(f'Dataset info: \n{len(dataset)}')
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_file_path, "a") as f:
        for i, batch in enumerate(batch_generator(dataset, batch_size=args.batch_size)):
            hasi = time.time()
            prompts = [
                prepare_example(question, answer)
                for question, answer in zip(batch['question'], batch['answer'])
            ]
            # print(prompts)
            outputs = [
                output.outputs[0].text.strip()
                for output in llm.chat(
                    prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            ]
            for split_name, index, output in zip(batch["split_name"], batch["index"], outputs):
                if not compiled_rule.search(output):
                    logger.error(f"Output does not match regex: {output}")
                    continue
                
                question, answer = postprocess_output(output)
                output_dict = {
                    "split_name": split_name,
                    "index": index,
                    "question": question,
                    "answer": answer,
                }
                print(output_dict)
                try:
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
                except UnicodeEncodeError: # Emojis raise this error
                    logger.error("Failed to encode output")
                    logger.error(output)
                    output_dict = {"conversation_id": index}
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
            logger.warning(f"Processed {progress + i * args.batch_size} examples, took {time.time()-hasi}seconds")
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="The HF model id.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompt.j2",
        help="The prompt.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="HiTZ/Magpie-Llama-3.70B-Instruct-Filtered",
        help="The dataset path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/scratch/asudupe/datasets/translation/",
        help="The output path.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.15)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
