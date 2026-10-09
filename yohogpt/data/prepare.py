"""Filter, deduplicate, decontaminate, FIM-transform and tokenize raw documents
into flat binary token shards ready for training.

    yoho prepare --raw data/raw/pretrain/starcoderdata --out data/processed/starcoderdata \
        --tokenizer artifacts/tokenizer/tokenizer.json --fim-rate 0.5 --workers 16

Output: ``<out>/tokens-00000.bin`` (uint16 if vocab < 65536 else uint32), each
document followed by <|endoftext|>, plus ``<out>/meta.json`` with statistics.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .dedup import Decontaminator, ExactDeduper, MinHashDeduper
from .filters import language_of, quality_reject_reason
from .fim import fim_transform, strip_specials
from .io import iter_records

_TOKENIZER = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    from ..tokenizer import YohoTokenizer

    _TOKENIZER = YohoTokenizer.load(tokenizer_path)


def _encode(args: tuple[str, bool]) -> list[int]:
    text, has_fim = args
    ids = _TOKENIZER.encode(text, allow_special=has_fim)
    ids.append(_TOKENIZER.eos_id)
    return ids


def load_eval_texts(paths: Iterable[str]) -> Iterator[str]:
    """Eval sets as JSONL with any of prompt/canonical_solution/test/text fields."""
    for p in paths:
        for rec in iter_records(p):
            for k in ("prompt", "canonical_solution", "test", "text", "question", "code", "problem_statement"):
                if isinstance(rec.get(k), str):
                    yield rec[k]


def filtered_docs(
    raw: Path,
    stats: Counter,
    near_dedup: bool = True,
    decontam: Decontaminator | None = None,
    fim_rate: float = 0.0,
    seed: int = 0,
) -> Iterator[tuple[str, bool]]:
    rng = random.Random(seed)
    exact = ExactDeduper()
    near = MinHashDeduper() if near_dedup else None
    for rec in iter_records(raw):
        text = rec.get("text", "")
        meta = rec.get("meta", {})
        stats["seen"] += 1
        path = meta.get("path") or meta.get("max_stars_repo_path") or ""
        lang = meta.get("lang") or (language_of(path) if path else None)
        reason = quality_reject_reason(text, lang)
        if reason:
            stats[f"drop_{reason}"] += 1
            continue
        if exact.is_duplicate(text):
            stats["drop_exact_dup"] += 1
            continue
        if near is not None and near.is_duplicate(text):
            stats["drop_near_dup"] += 1
            continue
        if decontam is not None and decontam.is_contaminated(text):
            stats["drop_contaminated"] += 1
            continue
        text = strip_specials(text)
        is_code = lang not in (None, "text", "markdown")
        if is_code and fim_rate > 0:
            new = fim_transform(text, rng, fim_rate)
            has_fim = new is not text
            text = new
        else:
            has_fim = False
        stats["kept"] += 1
        yield text, has_fim


class TokenShardWriter:
    def __init__(self, out_dir: Path, vocab_size: int, shard_tokens: int = 1 << 28) -> None:
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = np.uint16 if vocab_size < 65536 else np.uint32
        self.shard_tokens = shard_tokens
        self.buf: list[np.ndarray] = []
        self.buf_len = 0
        self.shard = 0
        self.total = 0

    def add(self, ids: list[int]) -> None:
        arr = np.asarray(ids, dtype=self.dtype)
        self.buf.append(arr)
        self.buf_len += len(arr)
        self.total += len(arr)
        if self.buf_len >= self.shard_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        np.concatenate(self.buf).tofile(self.out_dir / f"tokens-{self.shard:05d}.bin")
        self.shard += 1
        self.buf, self.buf_len = [], 0


def prepare(
    raw: Path,
    out: Path,
    tokenizer_path: str,
    fim_rate: float = 0.5,
    near_dedup: bool = True,
    eval_files: list[str] | None = None,
    workers: int = 4,
    shard_tokens: int = 1 << 28,
    seed: int = 0,
) -> dict:
    from ..tokenizer import YohoTokenizer

    tok = YohoTokenizer.load(tokenizer_path)
    decontam = Decontaminator(load_eval_texts(eval_files)) if eval_files else None
    stats: Counter = Counter()
    writer = TokenShardWriter(out, tok.vocab_size, shard_tokens)
    docs = filtered_docs(raw, stats, near_dedup, decontam, fim_rate, seed)

    if workers > 1:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
            for ids in pool.imap(_encode, docs, chunksize=64):
                writer.add(ids)
    else:
        _init_worker(tokenizer_path)
        for d in docs:
            writer.add(_encode(d))
    writer.flush()

    meta = {
        "tokens": writer.total,
        "shards": writer.shard,
        "dtype": np.dtype(writer.dtype).name,
        "vocab_size": tok.vocab_size,
        "stats": dict(stats),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), file=sys.stderr)
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--fim-rate", type=float, default=0.5)
    ap.add_argument("--no-near-dedup", action="store_true")
    ap.add_argument("--eval-files", nargs="*", default=[], help="JSONL eval sets to decontaminate against")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--shard-tokens", type=int, default=1 << 28)
    args = ap.parse_args(argv)
    prepare(Path(args.raw), Path(args.out), args.tokenizer, args.fim_rate, not args.no_near_dedup,
            args.eval_files, args.workers, args.shard_tokens)


if __name__ == "__main__":
    main()
