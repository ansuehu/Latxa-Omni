"""Translate ICTNLP/InstructS2S-200K (prepared as JSONL by prepare_instructs2s.py)
from English to Basque with Latxa, keeping each conversation's turns intact.

This is a sibling of translate.py, not a rewrite of it: that script's schema (flat
`question`/`answer` columns, one exchange per row) and its resume/parsing logic
share almost nothing with the nested `{"id", "turns": [...]}` schema here, and
translate.py is the provenance record for the already-produced
VoiceAssistant-400K_eu -- there is no reason to disturb it.

Design choices:

  * One vLLM request per *conversation*, not per turn. Turn 2+ of these
    dialogues is full of pronouns and ellipsis that only the earlier turns
    resolve ("Bai, zein ikasgai da?" is meaningless alone) -- per-turn
    requests would translate each line in isolation and lose that.
  * Numbered plain-text markers ("ERABILTZAILEA 1: ...", "LAGUNTZAILEA 1:
    ...") instead of JSON/guided decoding. Turn count varies per
    conversation, so a JSON schema pinning it would have to be rebuilt per
    length, and structured-output count guarantees are backend-dependent
    anyway -- you'd still need this script's own count check. Plain text also
    avoids forcing the model to escape quotes inside Basque prose.
  * Resume by the *set of conversation ids already written*, not by counting
    output lines: a row that fails parsing is retried, not silently
    skipped-and-miscounted (translate.py's bug).
  * Never write a partial/stub record: a conversation is written once, in
    full, or not at all (its id goes to <output>.failed.txt instead).

Usage:
    python translate_instructs2s.py --selftest
    python translate_instructs2s.py --jsonl_path instruct_en_200k.jsonl \
        --dry_run --limit 5
    python translate_instructs2s.py --jsonl_path instruct_en_200k.jsonl \
        --output_path /scratch/asudupe/datasets/InstructS2S-200K/eu \
        --model_path HiTZ/Latxa-Llama-3.1-8B-Instruct --limit 200
"""
from argparse import ArgumentParser
import json
import logging
import os
import re
import sys
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --------------------------------------------------------------------------
# Turn numbering / marker format shared by the prompt builder and the parser
# --------------------------------------------------------------------------

# Labels used when *presenting* the English source to the model.
_SOURCE_ROLE_TAG = {"human": "erabiltzailea", "gpt": "laguntzailea"}
# Labels the model must use in its Basque output. Deliberately different
# casing/wording from the source tags above so the model cannot get away
# with copying the English framing verbatim into its answer.
_OUTPUT_ROLE_LABEL = {"human": "ERABILTZAILEA", "gpt": "LAGUNTZAILEA"}

# A turn's "exchange number" is 1 for the first human/gpt pair, 2 for the
# second, etc. -- both turns of a pair share it (matches how a human would
# refer to "the second exchange" of a conversation).
def turn_labels(turns):
    exchange = 0
    labels = []
    for turn in turns:
        if turn["from"] == "human":
            exchange += 1
        labels.append(exchange)
    return labels


def build_source_block(turns):
    labels = turn_labels(turns)
    lines = [
        f"{n} ({_SOURCE_ROLE_TAG[turn['from']]}): {turn['text']}"
        for turn, n in zip(turns, labels)
    ]
    return "\n".join(lines)


def expected_output_sequence(turns):
    labels = turn_labels(turns)
    return [(_OUTPUT_ROLE_LABEL[turn["from"]], n) for turn, n in zip(turns, labels)]


# Anchored at the start of a line ("^" with re.M) so a marker-shaped string
# that a translated turn merely happens to contain mid-sentence cannot be
# mistaken for the start of the next turn -- it would have to appear at the
# very start of a physical line to do that, and even then the strict
# (role, n) sequence check below would almost certainly catch it.
_TURN_RE = re.compile(
    r"^(ERABILTZAILEA|LAGUNTZAILEA)[ \t]+(\d+):[ \t]*(.*?)(?=^(?:ERABILTZAILEA|LAGUNTZAILEA)[ \t]+\d+:|\Z)",
    re.M | re.S,
)


def parse_translation(output_text, turns):
    """Parse a model response against the turns it was asked to translate.

    Returns (translated_turns, None) on success, or (None, reason) on
    failure. translated_turns is a list of {"from", "text"} in the same
    order as `turns`, ready to be written out.
    """
    matches = _TURN_RE.findall(output_text)
    if not matches:
        return None, "no ERABILTZAILEA/LAGUNTZAILEA markers found"

    parsed_seq = [(role, int(n)) for role, n, _ in matches]
    expected_seq = expected_output_sequence(turns)
    if parsed_seq != expected_seq:
        return None, f"turn sequence mismatch: expected {expected_seq}, got {parsed_seq}"

    translated_turns = [
        {"from": turn["from"], "text": text.strip()}
        for turn, (_, _, text) in zip(turns, matches)
    ]
    if any(not t["text"] for t in translated_turns):
        return None, "a translated turn is empty"
    return translated_turns, None


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a helpful AI assistant that specializes in English to Basque translation of spoken-style conversations.
Your task is to translate multi-turn speech-to-speech conversations from English to Basque, turn by turn.

Guidelines:
1. Maintain the original meaning and intent of every turn.
2. If a turn is about something culturally English/American, replace it with an equivalent Basque-culture reference (for example, a London landmark with a Bilbao one, an English band with a Basque one). Keep such substitutions consistent across every turn of the same conversation: if you localize a place or name in one turn, use that same localization in every later turn that refers back to it.
3. If a turn relies on an English idiom, joke, wordplay, or culture-specific expression, replace it with an equivalent Basque expression rather than translating literally.
4. These are transcripts of spoken conversation: keep filler words and disfluencies ("so,", "um,", "you know", "like") as natural Basque fillers ("beno,", "ba,", "badakizu", "e...") instead of cleaning them into tidy written Basque. This spoken register is the entire point of this dataset -- do not remove it.
5. Use standard Basque (batua).
6. Keep technical terms that do not have a widely accepted Basque translation as they are.
7. If a turn refers to the assistant as "Omni", use "Latxa-Omni" instead.
8. If a turn is specifically about the spelling, grammar, or wording of an English sentence (for example "check this sentence for mistakes: ..."), keep that English sentence quoted and unchanged, and translate only the surrounding instruction and explanation into Basque -- translating the English example itself would make the task meaningless.
9. If a turn asks for a sentence, phrase, or paragraph to be built around specific given words (for example "make a sentence using the words purchase, online and store"), translate those target words into Basque too, and use the same Basque words in both the request and the answer. Do not leave the English word quoted in the request: every turn here is read aloud by a Basque voice, so a stray English word has no correct pronunciation. Only keep a word in its original form when it genuinely has no Basque equivalent (guideline 6).
10. Translate every turn. Never add, drop, merge, or reorder turns. Reproduce the exact same numbering you were given, and use exactly this output format -- one line per turn, nothing before the first line and nothing after the last:
ERABILTZAILEA <n>: <translated turn>
LAGUNTZAILEA <n>: <translated turn>
"""

_FEWSHOT = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "Translate the following conversation to Basque, exchange by exchange, "
            "keeping the numbering shown:\n\n"
            "1 (erabiltzailea): Can you help me with my homework?\n"
            "1 (laguntzailea): Sure, what subject is it?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "ERABILTZAILEA 1: Lagundu ahal didazu nire etxeko lanekin?\n"
            "LAGUNTZAILEA 1: Bai, noski, zein ikasgai da?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Translate the following conversation to Basque, exchange by exchange, "
            "keeping the numbering shown:\n\n"
            "1 (erabiltzailea): So, um, what's the most emblematic thing in London?\n"
            "1 (laguntzailea): The most emblematic thing in London is Big Ben.\n"
            "2 (erabiltzailea): Is it very popular with tourists?\n"
            "2 (laguntzailea): Yes, it attracts millions of visitors every year."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "ERABILTZAILEA 1: Beno, ba, zein da gauzarik enblematikoena Bilbon?\n"
            "LAGUNTZAILEA 1: Bilboko gauzarik enblematikoena Guggenheim Museoa da.\n"
            "ERABILTZAILEA 2: Turista asko erakartzen al ditu?\n"
            "LAGUNTZAILEA 2: Bai, milioika bisitari erakartzen ditu urtero."
        ),
    },
    {
        "role": "user",
        "content": (
            "Translate the following conversation to Basque, exchange by exchange, "
            "keeping the numbering shown:\n\n"
            "1 (erabiltzailea): What is your name?\n"
            "1 (laguntzailea): I'm Omni, your voice assistant."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "ERABILTZAILEA 1: Zein da zure izena?\n"
            "LAGUNTZAILEA 1: Latxa-Omni naiz, zure ahotsezko laguntzailea."
        ),
    },
    {
        "role": "user",
        "content": (
            "Translate the following conversation to Basque, exchange by exchange, "
            "keeping the numbering shown:\n\n"
            "1 (erabiltzailea): Check this sentence for mistakes: He finnished his meal and left the resturant.\n"
            "1 (laguntzailea): The sentence contains two errors. The correct sentence should read: "
            "He finished his meal and left the restaurant."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "ERABILTZAILEA 1: Egiaztatu esaldi honek akatsik duen: He finnished his meal and left the resturant.\n"
            "LAGUNTZAILEA 1: Esaldiak bi akats ditu. Esaldi zuzenak honela jarri behar du: "
            "He finished his meal and left the restaurant."
        ),
    },
    {
        "role": "user",
        "content": (
            "Translate the following conversation to Basque, exchange by exchange, "
            "keeping the numbering shown:\n\n"
            "1 (erabiltzailea): Hey, can you make a sentence using the words purchase, online and store?\n"
            "1 (laguntzailea): You can purchase products online at our store."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "ERABILTZAILEA 1: Aizu, esaldi bat egin dezakezu erosi, online eta denda hitzak erabiliz?\n"
            "LAGUNTZAILEA 1: Produktuak online eros ditzakezu gure dendan."
        ),
    },
]


def build_prompt(turns):
    user_content = (
        "Translate the following conversation to Basque, exchange by exchange, "
        "keeping the numbering shown:\n\n" + build_source_block(turns)
    )
    return _FEWSHOT + [{"role": "user", "content": user_content}]


# --------------------------------------------------------------------------
# I/O: reading the sharded input, resuming from the output
# --------------------------------------------------------------------------

def iter_shard_records(jsonl_path, shard_idx, num_shards, limit):
    n_yielded = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no % num_shards != shard_idx:
                continue
            if limit is not None and n_yielded >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n_yielded += 1


def load_done_ids(output_path):
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Truncated last line from a job that was killed mid-write.
                continue
            if "id" in rec:
                done.add(rec["id"])
    return done


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def prompt_too_long(tokenizer, messages, max_tokens, max_model_len):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Depending on the transformers version, tokenize=True returns either a
    # plain list of token ids, or a dict-like BatchEncoding with an
    # "input_ids" key (BatchEncoding is not a `dict` subclass, so
    # isinstance(encoded, dict) would miss it -- try the key instead).
    try:
        token_ids = encoded["input_ids"]
    except (TypeError, KeyError, IndexError):
        token_ids = encoded
    return len(token_ids) + max_tokens > max_model_len


# --------------------------------------------------------------------------
# Selftest -- pure Python, no vLLM, no GPU. Run with --selftest.
# --------------------------------------------------------------------------

def run_selftest():
    cases = [
        (
            "single exchange, success",
            [{"from": "human", "text": "a"}, {"from": "gpt", "text": "b"}],
            "ERABILTZAILEA 1: eus a\nLAGUNTZAILEA 1: eus b",
            True,
        ),
        (
            "three exchanges, success",
            [
                {"from": "human", "text": "a1"}, {"from": "gpt", "text": "b1"},
                {"from": "human", "text": "a2"}, {"from": "gpt", "text": "b2"},
                {"from": "human", "text": "a3"}, {"from": "gpt", "text": "b3"},
            ],
            "ERABILTZAILEA 1: x1\nLAGUNTZAILEA 1: y1\n"
            "ERABILTZAILEA 2: x2\nLAGUNTZAILEA 2: y2\n"
            "ERABILTZAILEA 3: x3\nLAGUNTZAILEA 3: y3",
            True,
        ),
        (
            "marker text embedded mid-line must not split the turn",
            [{"from": "human", "text": "a"}, {"from": "gpt", "text": "b"}],
            "ERABILTZAILEA 1: Norbaitek esan zuen ERABILTZAILEA 2: bezala hitz egiten zuela\n"
            "LAGUNTZAILEA 1: erantzuna",
            True,
        ),
        (
            "emoji in translated text",
            [{"from": "human", "text": "a"}, {"from": "gpt", "text": "b"}],
            "ERABILTZAILEA 1: kaixo \U0001F600\nLAGUNTZAILEA 1: kaixo zuri ere",
            True,
        ),
        (
            "missing final turn is rejected",
            [{"from": "human", "text": "a"}, {"from": "gpt", "text": "b"}],
            "ERABILTZAILEA 1: eus a",
            False,
        ),
        (
            "extra hallucinated turn is rejected",
            [{"from": "human", "text": "a"}, {"from": "gpt", "text": "b"}],
            "ERABILTZAILEA 1: eus a\nLAGUNTZAILEA 1: eus b\nERABILTZAILEA 2: gehiegi",
            False,
        ),
    ]

    failures = 0
    for name, turns, output_text, should_succeed in cases:
        result, reason = parse_translation(output_text, turns)
        ok = (result is not None) == should_succeed
        status = "OK  " if ok else "FAIL"
        detail = "" if ok else f" (expected {'success' if should_succeed else 'failure'}, reason={reason!r})"
        print(f"{status} {name}{detail}")
        if not ok:
            failures += 1
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return failures == 0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_dry_run(jsonl_path, limit):
    """Print the exact prompt that would be sent, for a few conversations of
    different lengths -- no model load, so this runs in seconds."""
    n_shown = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and n_shown >= limit:
                return
            rec = json.loads(line)
            messages = build_prompt(rec["turns"])
            print(f"=== {rec['id']} ({len(rec['turns']) // 2} exchanges) ===")
            for m in messages:
                print(f"--- {m['role']} ---")
                print(m["content"])
            print()
            n_shown += 1


def translate_batch(llm, sampling_params, records):
    """Run one vLLM call over `records` and return (successes, failures),
    where successes is a list of (record, translated_turns) and failures is
    a list of (record, reason)."""
    prompts = [build_prompt(rec["turns"]) for rec in records]
    outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=False)
    successes, failures = [], []
    for rec, output in zip(records, outputs):
        text = output.outputs[0].text.strip()
        translated_turns, reason = parse_translation(text, rec["turns"])
        if translated_turns is None:
            failures.append((rec, reason))
        else:
            successes.append((rec, translated_turns))
    return successes, failures


def translate_conversation_per_exchange(llm, sampling_params, rec):
    """Last-resort fallback: translate each exchange of `rec` independently
    (no cross-exchange context). Used only after both the whole-conversation
    attempt and its retry failed to parse. This trades away pronoun/context
    coherence across exchanges for a much simpler, much more reliable
    per-request format (always "exchange 1" of a 1-exchange conversation),
    so it should recover the vast majority of otherwise-lost conversations.
    Returns (translated_turns, None) or (None, reason)."""
    turns = rec["turns"]
    translated_turns = []
    for i in range(0, len(turns), 2):
        pair = turns[i : i + 2]
        prompts = [build_prompt(pair)]
        outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=False)
        text = outputs[0].outputs[0].text.strip()
        pair_translated, reason = parse_translation(text, pair)
        if pair_translated is None:
            return None, f"exchange {i // 2 + 1}: {reason}"
        translated_turns.extend(pair_translated)
    return translated_turns, None


def main(args):
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if not args.jsonl_path:
        sys.exit("--jsonl_path is required (unless --selftest)")

    if args.dry_run:
        run_dry_run(args.jsonl_path, args.limit)
        return

    if not args.output_path:
        sys.exit("--output_path is required (unless --dry_run/--selftest)")

    # Imported here, not at module load, so --selftest and --dry_run work
    # without vLLM installed -- useful for iterating on the prompt/parser
    # without a GPU allocation.
    from vllm import LLM, SamplingParams

    out_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    shard_base = f"{args.output_path}.shard{args.shard_idx:02d}"
    output_path = f"{shard_base}.jsonl"
    failed_path = f"{shard_base}.failed.txt"
    toolong_path = f"{shard_base}.toolong.jsonl"

    done_ids = load_done_ids(output_path)
    logger.info(f"{len(done_ids)} conversations already translated in {output_path}; resuming.")

    records = [
        rec
        for rec in iter_shard_records(args.jsonl_path, args.shard_idx, args.num_shards, args.limit)
        if rec["id"] not in done_ids
    ]
    logger.info(f"{len(records)} conversations left to translate in shard {args.shard_idx}/{args.num_shards}.")
    if not records:
        return

    logger.info("Loading vLLM model...")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    greedy_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )
    retry_params = SamplingParams(
        temperature=0.7,
        seed=1,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    # Pre-flight length filter: divert conversations whose rendered prompt
    # (few-shot + this conversation) would exceed the context window instead
    # of letting vLLM silently truncate or error mid-batch.
    fits, toolong = [], []
    for rec in records:
        if prompt_too_long(tokenizer, build_prompt(rec["turns"]), args.max_tokens, args.max_model_len):
            toolong.append(rec)
        else:
            fits.append(rec)
    if toolong:
        logger.warning(f"{len(toolong)} conversations exceed max_model_len - max_tokens; diverted to {toolong_path}")
        with open(toolong_path, "a", encoding="utf-8") as f:
            for rec in toolong:
                print(json.dumps(rec, ensure_ascii=False), file=f)

    n_ok = n_retried = n_fallback = n_failed = 0

    with open(output_path, "a", encoding="utf-8") as out_f, open(failed_path, "a", encoding="utf-8") as fail_f:
        for batch_no, batch in enumerate(chunked(fits, args.batch_size)):
            t0 = time.time()
            successes, failures = translate_batch(llm, greedy_params, batch)

            if failures:
                retry_records = [rec for rec, _ in failures]
                retry_successes, retry_failures = translate_batch(llm, retry_params, retry_records)
                n_retried += len(retry_successes)
                successes.extend(retry_successes)
                failures = retry_failures

            for rec, translated_turns in successes:
                out_f.write(json.dumps({"id": rec["id"], "turns": translated_turns}, ensure_ascii=False) + "\n")
            out_f.flush()
            n_ok += len(successes)

            for rec, reason in failures:
                translated_turns, fallback_reason = translate_conversation_per_exchange(llm, greedy_params, rec)
                if translated_turns is not None:
                    out_f.write(json.dumps({"id": rec["id"], "turns": translated_turns}, ensure_ascii=False) + "\n")
                    n_fallback += 1
                else:
                    fail_f.write(f"{rec['id']}\twhole-conversation: {reason}\tper-exchange: {fallback_reason}\n")
                    n_failed += 1
            out_f.flush()
            fail_f.flush()

            logger.info(
                f"batch {batch_no}: {len(batch)} conversations in {time.time() - t0:.1f}s "
                f"(running totals: {n_ok} ok, {n_retried} retried, {n_fallback} per-exchange fallback, {n_failed} failed)"
            )

    logger.info(
        f"Done. {n_ok} translated on first attempt, {n_retried} after retry, "
        f"{n_fallback} via per-exchange fallback, {n_failed} failed "
        f"(see {failed_path}), {len(toolong)} too long (see {toolong_path})."
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl_path", type=str, help="Output of prepare_instructs2s.py")
    parser.add_argument("--output_path", type=str, help="Base path; shard suffix + .jsonl is appended")
    parser.add_argument("--model_path", type=str, default="HiTZ/Latxa-Llama-3.1-70B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=2048)
    # Measured against the full filtered corpus: prompt length (few-shot +
    # conversation, tokenized) tops out at 1919 tokens, p99 = 1425. 6144
    # (prompt + --max_tokens generation, ~2x headroom over the true max) is
    # nowhere near the 25700 the original VoiceAssistant-400K translate.py
    # used -- copying that value here left too little free GPU memory for
    # vLLM's KV cache with the 70B model on 2 GPUs (weights alone take
    # ~65.7 GiB/GPU), and the engine refused to start.
    parser.add_argument("--max_model_len", type=int, default=6144)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Only translate the first N conversations of this shard")
    parser.add_argument("--dry_run", action="store_true", help="Print prompts for --limit conversations and exit; no model load")
    parser.add_argument("--selftest", action="store_true", help="Run the parser's fixed test cases and exit; no model load")
    args = parser.parse_args()
    main(args)
