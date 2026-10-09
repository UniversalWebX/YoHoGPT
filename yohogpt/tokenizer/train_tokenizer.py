"""Train the YoHoGPT byte-level BPE tokenizer on a sample of the data mix.

Code-aware choices:
  * byte-level BPE -> no unknown tokens for any language or binary-ish text
  * digits split individually -> better arithmetic / version-number handling
  * whitespace runs kept as tokens -> indentation in Python/YAML is cheap
  * ChatML, FIM and tool-call special tokens reserved up front

Usage:
    python -m yohogpt.tokenizer.train_tokenizer --input data/raw --out artifacts/tokenizer --vocab-size 49152
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Iterator

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from .tokenizer import SPECIAL_TOKENS, YohoTokenizer

# GPT-4-style split pattern, tuned so runs of spaces/tabs (indentation) stay together.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|[ \t]+(?!\S)|[ \t]+|\s+"""
)


def iter_jsonl_texts(paths: list[Path], max_chars: int, seed: int = 0) -> Iterator[str]:
    """Yield document texts from .jsonl / .jsonl.gz / plain text files until max_chars."""
    rng = random.Random(seed)
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and f.suffix in {".jsonl", ".gz", ".txt"}]
        else:
            files.append(p)
    rng.shuffle(files)
    total = 0
    for f in files:
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if f.suffix == ".txt":
                    text = line
                else:
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                if not text:
                    continue
                text = text[:100_000]  # cap pathological docs
                total += len(text)
                yield text
                if total >= max_chars:
                    return


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train(texts: Iterator[str], vocab_size: int, out_dir: Path, min_frequency: int = 2) -> YohoTokenizer:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    return YohoTokenizer(tok)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="files/dirs of .jsonl(.gz) or .txt")
    ap.add_argument("--out", default="artifacts/tokenizer")
    ap.add_argument("--vocab-size", type=int, default=49152)
    ap.add_argument("--max-chars", type=int, default=5_000_000_000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args(argv)

    texts = iter_jsonl_texts([Path(p) for p in args.input], args.max_chars)
    tk = train(texts, args.vocab_size, Path(args.out), args.min_frequency)
    sample = "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
    ids = tk.encode(sample)
    print(f"vocab={tk.vocab_size} sample tokens={len(ids)} chars/token={len(sample) / len(ids):.2f}")
    assert tk.decode(ids) == sample, "round-trip failed"


if __name__ == "__main__":
    main()
