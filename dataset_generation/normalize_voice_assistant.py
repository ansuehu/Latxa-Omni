import argparse
import os
import re
import subprocess
import sys
import unicodedata

from datasets import DatasetDict, load_from_disk

_script_dir = os.path.dirname(os.path.abspath(__file__))
_BINARY = os.path.join(_script_dir, "vits", "modulo1y2")
# The complete Basque normalization dictionary is ahoNT/dicts/eu_dicc.dic
# (867,863 bytes) -- it is also modulo1y2's own default HDic name. The
# similarly-named vits/dict/eu_dic.dic (768,761 bytes) is missing thousands
# of entries: using it makes modulo1y2 print
# "[aHoTTS warn]: Entry too long in HDic dictionary" for most inputs and
# silently fall back to spelling numbers out digit-by-digit, e.g.
# "1937an" -> "bat bederatzi hiru zazpi an" instead of the correct
# "mila bederatziehun eta hogeita hamazazpian". Both the CLI binary here and
# the ctypes/.so binding in ahoNT/ahoNT.py read the same dictionary *data*;
# they are not different dictionaries for different bindings.
_DICT = os.path.join(_script_dir, "ahoNT", "dicts", "eu_dicc")

# modulo1y2 reads a well-formed Roman numeral immediately followed by "." as
# an ordinal (e.g. "XX. mendean" -> "hogeigarren mendean", correct), but also
# reads a bare well-formed Roman numeral as a cardinal/ordinal number even
# when it is actually an acronym: "DC-k" -> "seiehungarrenek" ("600th-erg"),
# "CD-a" -> "laurehungarrena" ("400th"), "MI-ren" -> "mila eta batgarreneren"
# ("1001st"). Non-well-formed sequences of the same letters are unaffected
# ("DVD" -> "de uve de", "LCD" -> "ele ze de" -- both already correct),
# so we only need to break up genuinely valid Roman numerals.
_ROMAN_RE = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,})\b(?!\.)")

# Curly/typographic punctuation that either has no ISO-8859-15 codepoint (and
# so become "?" once encoded with errors="replace") or that modulo1y2 does
# not treat like its ASCII equivalent.
_UNICODE_MAP = {
    "‘": "'", "’": "'",  # ' '
    "“": "", "”": "",  # " " (matches clean_text() in vits/inference_andoni.py, which drops '"' entirely)
    "–": "-", "—": "-",  # en/em dash
    "…": "...",  # ellipsis
    " ": " ",  # nbsp
}

# Applied only right after a digit, longest unit first so e.g. "km" is
# consumed whole before the bare "m" rule can see it.
_UNIT_EXPANSIONS = [
    (re.compile(r"(?<=\d)\s?km\b"), " kilometro"),
    (re.compile(r"(?<=\d)\s?cm\b"), " zentimetro"),
    (re.compile(r"(?<=\d)\s?mm\b"), " milimetro"),
    (re.compile(r"(?<=\d)\s?kg\b"), " kilogramo"),
    (re.compile(r"(?<=\d)\s?mg\b"), " miligramo"),
    (re.compile(r"(?<=\d)\s?ml\b"), " mililitro"),
    (re.compile(r"(?<=\d)\s?GB\b"), " gigabyte"),
    (re.compile(r"(?<=\d)\s?MB\b"), " megabyte"),
    (re.compile(r"(?<=\d)\s?KB\b"), " kilobyte"),
    (re.compile(r"(?<=\d)\s?°C\b"), " gradu zentigradu"),
    (re.compile(r"(?<=\d)\s?°F\b"), " gradu fahrenheit"),
    (re.compile(r"(?<=\d)\s?m\b"), " metro"),
    (re.compile(r"(?<=\d)\s?g\b"), " gramo"),
    (re.compile(r"(?<=\d)\s?l\b"), " litro"),
]

# A hyphen joining two lowercase Basque word parts (a normal compound, e.g.
# "film-foroetan", "ahots-laguntzaile") makes modulo1y2 treat the whole
# thing as one opaque token: it neither declines nor expands the first part
# correctly ("film-foroetan" -> "film foroetan", not "filme foroetan" --
# "film" alone correctly expands to "filme"), and can even garble it
# ("film-gauean" was previously seen misread as "filn gauean" through the
# wrong dictionary; with the right dictionary it still fails to decline).
# Splitting the compound into two independent words before normalizing
# fixes this, since each half is then looked up separately and correctly.
_COMPOUND_HYPHEN_RE = re.compile(r"(?<=[a-zà-ÿ])-(?=[a-zà-ÿ])")

# A "/" between two words used the way English uses "or" (e.g.
# "arratsalde/iluntzeko" = "afternoon/evening") is read aloud as the literal
# word "barra" ("slash") rather than as a separator; digit/digit fractions
# are handled separately below and are unaffected by this (alpha-only) rule.
_WORD_SLASH_RE = re.compile(r"(?<=[a-zà-ÿ])/(?=[a-zà-ÿ])")

# A hyphen joining a short (<=4 letter) uppercase acronym straight to a
# following digit is read aloud as the literal word "gidoia" ("hyphen"):
# "SS-5" -> "ese ese gidoia bost", "F-16" -> "efe gidoia bat sei". Capping
# the run at 4 letters keeps this from touching a longer all-caps word like
# "COVID-19", which modulo1y2 already reads correctly as "Covid emeretzi"
# without any help; two digit runs ("2020-2021", "10-15") are also already
# read correctly and are unaffected (this pattern requires letters first).
_ACRONYM_DIGIT_HYPHEN_RE = re.compile(r"\b([A-Z]{1,4})-(?=\d)")

_FRACTION_EXPANSIONS = [
    (re.compile(r"\b1\s*/\s*2\b"), "erdi bat"),
    (re.compile(r"\b1\s*/\s*4\b"), "laurden bat"),
    (re.compile(r"\b3\s*/\s*4\b"), "hiru laurden"),
]

# "15garrena" -> "15.a", "3garrena" -> "3.a": modulo1y2 expands "N.<suffix>"
# as an ordinal ("hamabosgarrena", "hirugarrena") but reads the literal
# "garren" text as a separate word ("15garrena" -> "bat bost garren").
_ORDINAL_RE = re.compile(r"(\d+)garren(\w*)")

# A quoted song/album/movie title (very common in this dataset, e.g.
# '"Master of Puppets"') gets partially rewritten by modulo1y2's
# Basque-loanword dictionary word-by-word -- some words match an entry and
# get respelled, others don't, producing incoherent output like
# '"City of Evil"' -> "ziti of ebil" or '"Run to the Hills"' -> "Run to the
# jils". Capping the match at one line/80 chars keeps this from swallowing a
# genuinely long quoted *sentence* of dialogue, which is more likely to be
# real Basque speech that should still get number expansion, rather than a
# short foreign title.
_QUOTE_RE = re.compile(r'"([^"\n]{1,80})"')

# Placeholder used to stand in for a protected quoted span while the rest of
# pre_normalize's regexes run. "\x01" is a control character that doesn't
# match any character class used by those regexes ([a-zà-ÿ], \d, markdown
# punctuation, ...), so it passes through inert; it is split back out again
# before modulo1y2 ever sees it (see _run_modulo1y2_segmented), since
# modulo1y2 itself does NOT pass unrecognized tokens through unchanged (it
# reads a literal placeholder aloud letter-by-letter, e.g. spelling out
# "QQQ0QQQ", and can even truncate the rest of its output when fed a raw
# control character).
_PLACEHOLDER_RE = re.compile(r"\x01(\d+)\x01")


def _fix_unicode_punctuation(text):
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    return text


def _strip_markdown(text):
    text = re.sub(r"(?m)^[ \t]*[-*+][ \t]+", "", text)  # bullet list markers
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", text)  # headings
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"__(.+?)__", r"\1", text)  # __bold__
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)  # _italic_
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)  # *italic*
    # Leftover stray markdown characters modulo1y2 would otherwise verbalize
    # (e.g. "**" -> "izartxo izartxo", "_" -> "beheko gidoia", "#" -> "almohadilla").
    for ch in ("*", "_", "`", "#"):
        text = text.replace(ch, "")
    return text


def _normalize_punctuation(text):
    # Mirrors clean_text() in vits/inference_andoni.py, the same
    # preprocessing the answer audio itself was synthesized with -- except
    # ":" is left alone between two digits, where modulo1y2 already reads it
    # correctly as a time ("10:30ean" -> "goizeko hamarrak eta hogeita
    # hamarrean"); turning that ":" into "," breaks the time reading.
    text = re.sub(r"(?<!\d):(?!\d)", ",", text)
    for ch in (";", "(", ")"):
        text = text.replace(ch, ",")
    text = text.replace('"', "")
    return text


def _join_lines(text):
    # A bare newline is otherwise silently dropped by modulo1y2's own output
    # joining (it does not insert a pause or period), which welds the last
    # word of one line onto the first word of the next. A leading "1. "/"2)"
    # list marker is read as a cardinal number glued onto the next word
    # ("1. Lehena" -> "bat. Lehena" merges into "bat puntu Lehena"-like noise),
    # so it is stripped before rejoining.
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*\d+[.)]\s+", "", line).strip()
        if line:
            lines.append(line)
    joined = ""
    for line in lines:
        if joined:
            joined += " " if joined.endswith((".", "!", "?", ",", ":", ";")) else ". "
        joined += line
    return joined


def _protect_roman_acronyms(text):
    def repl(match):
        token = match.group(1)
        if _ROMAN_RE.match(token):
            return "-".join(token)
        return token

    return _ACRONYM_RE.sub(repl, text)


def _expand_units_and_fractions(text):
    for pattern, replacement in _UNIT_EXPANSIONS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _FRACTION_EXPANSIONS:
        text = pattern.sub(replacement, text)
    return text


def _protect_quotes(text):
    """Stash quoted spans behind an inert placeholder, to be restored after
    modulo1y2 runs (see _QUOTE_RE and _PLACEHOLDER_RE above)."""
    protected = []

    def repl(match):
        protected.append(match.group(1))
        return f"\x01{len(protected) - 1}\x01"

    text = _QUOTE_RE.sub(repl, text)
    return text, protected


def pre_normalize(text, protected):
    """Basque-specific fixups applied before handing text to modulo1y2.

    Each rule here addresses a case modulo1y2 mis-reads on its own; see the
    module-level comments next to each helper (and dataset_generation's plan
    notes) for the validated before/after pairs. `protected` is the list
    populated by _protect_quotes; quote-stashing must happen first, right
    after unicode cleanup, since _normalize_punctuation would otherwise
    strip the quote marks _QUOTE_RE relies on to find the spans.
    """
    text = _fix_unicode_punctuation(text)
    text, stashed = _protect_quotes(text)
    protected.extend(stashed)
    text = _strip_markdown(text)
    text = _normalize_punctuation(text)
    text = _COMPOUND_HYPHEN_RE.sub(" ", text)
    text = _ACRONYM_DIGIT_HYPHEN_RE.sub(r"\1 ", text)
    text = _WORD_SLASH_RE.sub(" edo ", text)
    text = _join_lines(text)
    text = _ORDINAL_RE.sub(r"\1.\2", text)
    text = _protect_roman_acronyms(text)
    text = _expand_units_and_fractions(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _call_modulo1y2(text):
    """Run one chunk of already pre_normalize'd text through modulo1y2 and
    decode its output. Must never be called with a placeholder in `text`
    (see _PLACEHOLDER_RE) -- modulo1y2 does not pass unrecognized tokens
    through unchanged, so a placeholder would come back mangled or worse."""
    if not text.strip():
        return ""
    try:
        proc = subprocess.run(
            [
                _BINARY,
                f"-HDic={_DICT}",
                "-Lang=eu",
                "-TxtMode=Word",
            ],
            # This modulo1y2 binary family expects Latin text encoded as
            # ISO-8859-15 for eu/es (see dataset_generation/vits/dict/trans.sh
            # and dataset_generation/ahoNT/ahoNT.py's normalizazioa(), which
            # use the same encoding on both the way in and the way out).
            # Encoding as UTF-8 here silently mangled accented Basque input
            # (n~, u"|, a', ...). Characters with no ISO-8859-15 codepoint at
            # all are dropped (errors="ignore") rather than turned into "?"
            # (errors="replace"), which modulo1y2 would otherwise read aloud.
            input=text.encode("iso-8859-15", errors="ignore"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if proc.returncode != 0:
            stderr_head = proc.stderr.decode("iso-8859-15", errors="replace").splitlines()[:1]
            print(
                f"modulo1y2 returned {proc.returncode}: {stderr_head}",
                file=sys.stderr,
            )
            return text
        result = proc.stdout.decode("iso-8859-15", errors="replace")
        # modulo1y2 prints one line per sentence with no separator of its
        # own; joining with "" (the previous behavior) welds the last word
        # of one sentence onto the first word of the next whenever a line
        # happens not to end in a trailing space.
        return re.sub(r"\s+", " ", " ".join(result.splitlines())).strip()
    except Exception as e:
        print(f"Normalization error: {e}", file=sys.stderr)
        return text


def _run_modulo1y2_segmented(text, protected):
    """Call _call_modulo1y2 on each piece of `text` around the placeholders
    left by _protect_quotes, splicing the original quoted text back in
    verbatim at each placeholder position."""
    pieces = _PLACEHOLDER_RE.split(text)
    out = []
    for i, piece in enumerate(pieces):
        if i % 2 == 1:
            out.append(protected[int(piece)])
        else:
            normalized = _call_modulo1y2(piece)
            if normalized:
                out.append(normalized)
    result = " ".join(out)
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"\s+([,.!?:;])", r"\1", result)
    return result


def normalize_text(text):
    if not text:
        return ""
    protected = []
    text = pre_normalize(text, protected)
    if not text:
        return ""
    return _run_modulo1y2_segmented(text, protected)


def normalize_example(example):
    answer = example.get("answer", "")
    example["answer_normalized"] = normalize_text(answer)
    return example


_SELFTEST_CASES = [
    ("1937an", "mila bederatziehun eta hogeita hamazazpian"),
    ("1980ko", "mila bederatziehun eta laurogeiko"),
    ("25ean", "hogeita bostean"),
    ("2025eko", "bi mila eta hogeita bosteko"),
    ("15garrena", "hamabosgarrena"),
    ("XX. mendean", "hogeigarren mendean"),
    ("DC-k", "de zeka"),
    ("%25", "ehuneko hogeita bost"),
    ("25%", "ehuneko hogeita bost"),
    ("3,5 kg", "hiru koma bost kilogramo"),
    ("10:30ean", "goizeko hamarrak eta hogeita hamarrean"),
    ("2024/05/12", "bi mila eta hogeita lauko maiatzaren hamabia"),
    ("**Osagaiak**:", "Osagaiak,"),
    ("Iruñean", "Iruñean"),
    (
        'Talde klasikoak dira, hala nola "Paranoid" diskoagatik ezagunak.',
        "Talde klasikoak dira, hala nola Paranoid diskoagatik ezagunak.",
    ),
    (
        'Probatu "City of Evil" diskoarekin. 1995ean atera zen.',
        "Probatu City of Evil diskoarekin. mila bederatziehun eta laurogeita hamabostean atera zen.",
    ),
]


def run_selftest():
    failures = 0
    for text, expected in _SELFTEST_CASES:
        got = normalize_text(text)
        ok = got == expected
        print(f"{'OK  ' if ok else 'FAIL'} {text!r} -> {got!r} (expected {expected!r})")
        if not ok:
            failures += 1
    print(f"\n{len(_SELFTEST_CASES) - failures}/{len(_SELFTEST_CASES)} passed")
    return failures == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--dataset_out")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only normalize the first N rows of each split and print answer/"
        "answer_normalized side by side, without writing dataset_out.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run a fixed set of known-good normalization checks and exit.",
    )
    args = parser.parse_args()

    if args.selftest:
        if not os.path.isfile(_BINARY):
            print(f"modulo1y2 binary not found: {_BINARY}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(_DICT + ".dic"):
            print(f"Dictionary not found: {_DICT}.dic", file=sys.stderr)
            sys.exit(1)
        sys.exit(0 if run_selftest() else 1)

    if not args.dataset_path or (not args.dataset_out and not args.limit):
        parser.error("--dataset_path and --dataset_out are required (unless --selftest)")

    if args.dataset_out and os.path.abspath(args.dataset_out) == os.path.abspath(args.dataset_path):
        parser.error("--dataset_out must not be the same path as --dataset_path")

    if not os.path.isfile(_BINARY):
        print(f"modulo1y2 binary not found: {_BINARY}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(_DICT + ".dic"):
        print(f"Dictionary not found: {_DICT}.dic", file=sys.stderr)
        sys.exit(1)

    ds = load_from_disk(args.dataset_path)

    if args.limit:
        splits = ds.keys() if isinstance(ds, DatasetDict) else [None]
        for split in splits:
            data = ds[split] if split is not None else ds
            data = data.select(range(min(args.limit, len(data))))
            for ex in data:
                answer = ex.get("answer", "")
                normalized = normalize_text(answer)
                label = f"[{split}] " if split is not None else ""
                print(f"{label}ANSWER: {answer!r}")
                print(f"{label}NORM  : {normalized!r}")
                print()
        return

    if isinstance(ds, DatasetDict):
        for split in ds:
            ds[split] = ds[split].map(
                normalize_example, num_proc=args.num_proc, desc=f"Normalizing {split}"
            )
    else:
        ds = ds.map(normalize_example, num_proc=args.num_proc, desc="Normalizing")

    if args.validate:
        if isinstance(ds, DatasetDict):
            for split in ds:
                data = ds[split].select_columns(["answer", "answer_normalized"])
                changed = sum(
                    1 for ex in data
                    if ex.get("answer") and ex.get("answer_normalized") != ex.get("answer")
                )
                total = len(data)
                print(f"{split}: {changed}/{total} answers changed ({100 * changed / total:.1f}%)")
        else:
            data = ds.select_columns(["answer", "answer_normalized"])
            changed = sum(
                1 for ex in data
                if ex.get("answer") and ex.get("answer_normalized") != ex.get("answer")
            )
            total = len(data)
            print(f"dataset: {changed}/{total} answers changed ({100 * changed / total:.1f}%)")

    ds.save_to_disk(args.dataset_out)


if __name__ == "__main__":
    main()
