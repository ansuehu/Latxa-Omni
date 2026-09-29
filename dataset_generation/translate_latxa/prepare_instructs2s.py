"""Stream ICTNLP/InstructS2S-200K's source JSON into a compact, shardable JSONL.

The upstream file (`instruct_en_200k_data_cosy2.json`) is a single pretty-printed
JSON array, ~1 GB on disk. `json.load()`-ing it whole pulls the entire parsed
structure (plus Python object overhead) into RAM at once, and every downstream
script would have to re-pay that cost. Instead we stream it once with `ijson`
and write one compact JSON object per line, so every later stage (translation,
sharding, resume) is O(1) in memory and can `seek`/split by line number.

We also drop two fields from each turn while we're here:
  - "speech": a path to an *English* wav we don't have and won't use --
    Basque audio is re-synthesized from the translated text later.
  - "unit": CosyVoice2 speech tokens (values run up to ~6560). Latxa-Omni's
    speech generator was trained with `unit_vocab_size: 1000` (mHuBERT layer-9
    k-means units), a different, incompatible vocabulary. Units must be
    re-derived from Basque TTS output regardless, so the source ones are dead
    weight.

What's kept per conversation: `{"id": ..., "turns": [{"from": "human"|"gpt",
"text": ...}, ...]}`.

We also drop conversations whose task hinges on literal English (or other
non-Basque) text surviving verbatim -- checking a specific English sentence
for spelling/grammar mistakes, spelling an English word letter by letter, or
translating to/from a third language. These translate into Basque text just
fine, but the audio pipeline downstream (`vits/inference_andoni.py`) is a
Basque-only voice model: it phonemizes via the aholab Basque G2P system
(`getPhones()`/`modulo1y2`), which has no notion of English phonotactics, so
"He finnished his meal" or "p-n-e-u-m-o-n-i-a" would come out as mangled
Basque-read nonsense rather than mispronounced-but-recognizable English. A
scan of the full 200K found this affects roughly 0.3% of conversations (most
apparent "spelling/grammar" mentions are unrelated, e.g. "what mistakes did
Napoleon make" -- ordinary Basque-translatable content). Other
language-touching content (rhymes, synonyms, acronyms) is left alone: once
the target word is translated to Basque, asking for Basque synonyms/rhymes
of *that* word is perfectly synthesizable, no English text needs to survive.
Filtered whole-conversation (not per-turn), since a multi-turn conversation
with one turn excised would leave later turns referring to missing context.
Written to `<out>.language_dependent.jsonl` rather than silently dropped, so
they're reviewable (e.g. a future localization pass could rewrite rather
than discard them).

Usage:
    python prepare_instructs2s.py --src instruct_en_200k_data_cosy2.json \
        --out instruct_en_200k.jsonl
    python prepare_instructs2s.py --src ... --limit 2000 --stats
"""
import argparse
import json
import re
import sys
from collections import Counter

import ijson

# Tuned against the real dataset (see module docstring): precise enough that
# eyeballing dozens of matches per category turned up no false positives,
# while a naive "spelling|grammar|mistake" scan over-matched by ~9x on
# generic, perfectly-translatable uses of those words.
_LANGUAGE_DEPENDENT_PATTERNS = {
    "english_sentence_correction": re.compile(
        r"(check|evaluate|correct|fix|identify).{0,40}(sentence|paragraph).{0,40}(mistake|error|grammar|spelling)"
        r"|(sentence|paragraph).{0,40}(contains?|has).{0,20}(mistake|error)"
        r"|grammatically correct sentence"
        r"|correct(ed)? sentence (is|should|would)",
        re.I,
    ),
    "spell_the_word": re.compile(
        r"\bspell (the|this|that|out) word|how (do you|to) spell (the|this|that) word|is spelled [a-z]-[a-z]",
        re.I,
    ),
    "cross_language_translation": re.compile(
        r"translate.{0,40}(into|to|from) (french|spanish|german|italian|portuguese|japanese|chinese|korean|russian|arabic|basque|english)"
        r"|how do you say .{0,30} in (french|spanish|german|italian|portuguese|japanese|chinese|korean|russian|arabic)",
        re.I,
    ),
}


def find_language_dependent_reason(turns):
    """Return the name of the first matching pattern in
    _LANGUAGE_DEPENDENT_PATTERNS, or None. Checked against all turns joined
    together, so a match anywhere in the conversation excludes the whole
    thing (see module docstring for why)."""
    text = " ".join(t["text"] for t in turns)
    for name, pattern in _LANGUAGE_DEPENDENT_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def iter_conversations(src_path):
    with open(src_path, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


def to_compact_record(item):
    """Extract {id, turns:[{from, text}]} from a raw source item.

    Returns (record, problem) where problem is None on success, or a short
    string describing why the conversation was quarantined (record is then
    None).
    """
    conv_id = item.get("id")
    conversation = item.get("conversation")
    if not conv_id or not conversation:
        return None, "missing id or conversation"

    turns = []
    expected_from = "human"
    for turn in conversation:
        frm = turn.get("from")
        text = turn.get("text")
        if text is None:
            return None, f"turn missing text ({frm!r})"
        if frm != expected_from:
            return None, f"non-alternating turns (expected {expected_from!r}, got {frm!r})"
        turns.append({"from": frm, "text": text})
        expected_from = "gpt" if expected_from == "human" else "human"

    if not turns or turns[0]["from"] != "human" or turns[-1]["from"] != "gpt":
        return None, "conversation does not start with human / end with gpt"

    return {"id": conv_id, "turns": turns}, None


def run_stats(src_path, limit):
    n_turn_counts = Counter()
    n_quarantined = Counter()
    n_language_dependent = Counter()
    total = 0
    samples = []
    for item in iter_conversations(src_path):
        if limit is not None and total >= limit:
            break
        total += 1
        record, problem = to_compact_record(item)
        if problem is not None:
            n_quarantined[problem] += 1
            continue
        reason = find_language_dependent_reason(record["turns"])
        if reason is not None:
            n_language_dependent[reason] += 1
            continue
        n_pairs = len(record["turns"]) // 2
        n_turn_counts[n_pairs] += 1
        if len(samples) < 5:
            samples.append(record)

    print(f"Scanned {total} conversations.")
    print("\nExchange-count histogram (1 exchange = 1 human+gpt pair):")
    for n_pairs in sorted(n_turn_counts):
        print(f"  {n_pairs:2d} exchanges: {n_turn_counts[n_pairs]}")
    if n_quarantined:
        print("\nQuarantined (schema violations):")
        for reason, count in n_quarantined.most_common():
            print(f"  {count:6d}  {reason}")
    if n_language_dependent:
        total_ld = sum(n_language_dependent.values())
        print(f"\nExcluded (language-dependent, {100 * total_ld / total:.2f}% of scanned):")
        for reason, count in n_language_dependent.most_common():
            print(f"  {count:6d}  {reason}")
    print("\nSample conversations:")
    for s in samples:
        print(json.dumps(s, ensure_ascii=False, indent=2))


def run_convert(src_path, out_path, limit):
    written = 0
    quarantined = 0
    language_dependent = 0
    total = 0
    ld_path = f"{out_path}.language_dependent.jsonl"
    with open(out_path, "w", encoding="utf-8") as out_f, \
            open(ld_path, "w", encoding="utf-8") as ld_f:
        for item in iter_conversations(src_path):
            if limit is not None and total >= limit:
                break
            total += 1
            record, problem = to_compact_record(item)
            if problem is not None:
                quarantined += 1
                print(f"quarantined {item.get('id', '<no id>')}: {problem}", file=sys.stderr)
                continue
            reason = find_language_dependent_reason(record["turns"])
            if reason is not None:
                language_dependent += 1
                print(json.dumps({**record, "reason": reason}, ensure_ascii=False), file=ld_f)
                continue
            print(json.dumps(record, ensure_ascii=False), file=out_f)
            written += 1
    print(
        f"Wrote {written}/{total} conversations to {out_path} "
        f"({quarantined} quarantined, {language_dependent} excluded as "
        f"language-dependent -> see {ld_path})."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to instruct_en_200k_data_cosy2.json")
    parser.add_argument("--out", help="Output JSONL path (required unless --stats)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N conversations")
    parser.add_argument("--stats", action="store_true", help="Print turn-count stats and samples; don't write")
    args = parser.parse_args()

    if not args.stats and not args.out:
        parser.error("--out is required unless --stats is given")

    if args.stats:
        run_stats(args.src, args.limit)
    else:
        run_convert(args.src, args.out, args.limit)


if __name__ == "__main__":
    main()
