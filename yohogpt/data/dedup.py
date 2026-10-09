"""Exact and near-duplicate removal, plus benchmark decontamination.

Near-dedup uses MinHash + locality-sensitive hashing (bands x rows), the same
recipe used for The Stack / StarCoder, implemented with numpy only.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"\w+")
_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8", "replace")).hexdigest()


class ExactDeduper:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        h = content_hash(text)
        if h in self.seen:
            return True
        self.seen.add(h)
        return False


def _shingles(text: str, n: int) -> set[bytes]:
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < n:
        return {" ".join(toks).encode()}
    return {" ".join(toks[i : i + n]).encode() for i in range(len(toks) - n + 1)}


class MinHashDeduper:
    """Streaming near-duplicate detector.

    A document is a duplicate if it shares all ``rows`` minhash values with an
    earlier document in at least one of ``bands`` bands. With 16x8 the
    detection threshold is roughly Jaccard ~0.7 on 5-gram shingles.
    """

    def __init__(self, num_perm: int = 128, bands: int = 16, ngram: int = 5, seed: int = 42) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm, self.bands, self.rows, self.ngram = num_perm, bands, num_perm // bands, ngram
        rng = np.random.default_rng(seed)
        # a, b < 2^31 and x < 2^32 keep a*x + b below 2^64 (no uint64 overflow).
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)
        self.buckets: list[dict[bytes, int]] = [dict() for _ in range(bands)]
        self.count = 0

    def signature(self, text: str) -> np.ndarray:
        sh = _shingles(text, self.ngram)
        hv = np.fromiter(
            (int.from_bytes(hashlib.blake2b(s, digest_size=4).digest(), "little") for s in sh),
            dtype=np.uint64,
            count=len(sh),
        )
        phv = (hv[:, None] * self.a + self.b) % _MERSENNE & _MAX_HASH
        return phv.min(axis=0)

    def is_duplicate(self, text: str) -> bool:
        sig = self.signature(text)
        keys = [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]
        dup = any(k in self.buckets[i] for i, k in enumerate(keys))
        if not dup:
            for i, k in enumerate(keys):
                self.buckets[i][k] = self.count
        self.count += 1
        return dup


class Decontaminator:
    """Drop training documents that contain long n-gram overlaps with eval sets."""

    def __init__(self, eval_texts: Iterable[str], ngram: int = 13) -> None:
        self.ngram = ngram
        self.grams: set[int] = set()
        for t in eval_texts:
            self.grams.update(self._grams(t))

    def _grams(self, text: str) -> set[int]:
        toks = _TOKEN_RE.findall(text.lower())
        return {hash(tuple(toks[i : i + self.ngram])) for i in range(len(toks) - self.ngram + 1)}

    def is_contaminated(self, text: str) -> bool:
        if not self.grams:
            return False
        toks = _TOKEN_RE.findall(text.lower())
        n = self.ngram
        return any(hash(tuple(toks[i : i + n])) in self.grams for i in range(len(toks) - n + 1))


def dedup_stats(flags: Iterable[bool]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for f in flags:
        out["duplicate" if f else "kept"] += 1
    return dict(out)
