from argparse import ArgumentParser
import json
import os
from typing import List, Literal
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel, RootModel
import logging
from datasets import load_dataset
import jinja2 as j2
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
class Conversation(RootModel):
    root: List[Message]
examples = [
    [
        {
            "role": "user",
            "content": """Please translate the following text from Spanish to English: "El ayuntamiento de la ciudad de Barcelona se comprometió a permitir que los ciudadanos puedan acceder a la información pública a través de Internet. Diversas entidades y organizaciones decidieron colaborar con la ciudad para desarrollar un portal web que permita a los ciudadanos acceder a la información de la ciudad. Esta colaboración se denomina 'Portal de Participación Ciudadana' y el objetivo es proporcionar a los ciudadanos un fácil acceso a la información que les interesa y les afecta." Here is the text in Spanish. Please provide the accurate translation to English.""",
        },
        {
            "role": "assistant",
            "content": '''Here is the translation of the text from Spanish to English: "The city council of Barcelona has committed to allowing citizens to access public information through the Internet. Various entities and organizations have decided to collaborate with the city to develop a web portal that allows citizens to access city information. This collaboration is called the 'Citizen Participation Portal' and the goal is to provide citizens with easy access to the information that interests and affects them."''',
        },
    ]
]
def load_template(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        return j2.Template(f.read().strip())
def load_dataset_slice(dataset_path, slice=slice(0, None, 1)):
    dataset = load_dataset(dataset_path)
    return dataset["train"][slice]
def batch_generator(dataset, batch_size=1):
    keys = list(dataset.keys())
    values = list(zip(*dataset.values()))
    for i in range(0, len(values), batch_size):
        yield [
            {key: value for key, value in zip(keys, _values)}
            for _values in values[i : i + batch_size]
        ]
def from_sharegpt_to_chat_template(conversation):
    return [
        {"role": "user", "content": message["value"]}
        if message["from"] == "human"
        else {"role": "assistant", "content": message["value"]}
        for message in conversation
    ]
def prepare_example(example):
    return json.dumps(
        from_sharegpt_to_chat_template(example["conversations"]),
        ensure_ascii=False,
        indent=4,
    )
def postprocess_output(output):
    try:
        conversation = json.loads(output)
        return conversation
    except json.JSONDecodeError:
        ...
    # Try to fix the output by finding the last bracket and removing everything after it
    last_bracket = output.rfind("}")
    # If closing bracket is found, return empty list
    if last_bracket == -1:
        return []
    try:
        conversation = json.loads(output[: last_bracket + 1] + "]")
        return conversation
    except json.JSONDecodeError as e:
        logger.error("Failed to fix output")
        logger.error(e)
        logger.error(output)
        return []
def main(args):
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        stop="\n```",
        guided_decoding=GuidedDecodingParams(json=Conversation.model_json_schema()),
        frequency_penalty=args.frequency_penalty,
    )
    prompt_template = load_template(args.prompt_path)
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
    dataset = load_dataset_slice(
        args.dataset_path, slice(args.dataset_start + progress, args.dataset_end, 1)
    )
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_file_path, "a") as f:
        for i, batch in enumerate(batch_generator(dataset, batch_size=args.batch_size)):
            prompts = [
                prompt_template.render(english_conversation=prepare_example(example))
                for example in batch
            ]
            outputs = [
                output.outputs[0].text.strip()
                for output in llm.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True,
                )
            ]
            for original, output in zip(batch, outputs):
                conversation = postprocess_output(output)
                if len(conversation) < len(original["conversations"]):
                    output_dict = {"conversation_id": original["conversation_id"]}
                else:
                    # Make sure the generated conversation is the same length as the original
                    conversation = conversation[: len(original["conversations"])]
                    output_dict = {
                        "conversation_id": original["conversation_id"],
                        "instruction": original["instruction"],
                        "response": original["response"],
                        "conversations": from_sharegpt_to_chat_template(
                            original["conversations"]
                        ),
                        "translated_instruction": conversation[-2]["content"],
                        "translated_response": conversation[-1]["content"],
                        "translated_conversations": conversation,
                        **original,
                    }
                try:
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
                except UnicodeEncodeError: # Emojis raise this error
                    logger.error("Failed to encode output")
                    logger.error(output)
                    output_dict = {"conversation_id": original["conversation_id"]}
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
            logger.warning(f"Processed {progress + i * args.batch_size} examples")
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
        default="output/",
        help="The output path.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.15)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
