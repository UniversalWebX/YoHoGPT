"""Training datasets.

* ``PackedTokens``   - memory-mapped view over the ``tokens-*.bin`` shards of one source.
* ``MixtureSampler`` - samples fixed-length windows from several sources by weight
  (deterministic given seed/rank/step, so resumed runs see the same data).
* ``SFTDataset``     - chat conversations packed into sequences with assistant-only loss.
* ``PreferenceDataset`` - (prompt, chosen, rejected) triples for DPO.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from ..tokenizer import YohoTokenizer
from .io import iter_records


class PackedTokens:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        dtype = np.dtype(meta["dtype"])
        files = sorted(path.glob("tokens-*.bin"))
        if not files:
            raise FileNotFoundError(f"no token shards in {path}")
        self.shards = [np.memmap(f, dtype=dtype, mode="r") for f in files]
        self.sizes = np.array([len(s) for s in self.shards])
        self.total = int(self.sizes.sum())

    def sample(self, rng: np.random.Generator, length: int, holdout: float = 0.0, val: bool = False) -> np.ndarray:
        """Return ``length`` contiguous tokens from a random shard.

        The last ``holdout`` fraction of every shard is reserved for validation.
        """
        p = self.sizes / self.sizes.sum()
        for _ in range(100):
            i = rng.choice(len(self.shards), p=p)
            n = len(self.shards[i])
            split = int(n * (1 - holdout))
            lo, hi = (split, n) if val else (0, split)
            if hi - lo > length:
                start = int(rng.integers(lo, hi - length))
                return np.asarray(self.shards[i][start : start + length], dtype=np.int64)
        raise ValueError(f"shards too small for windows of {length} tokens")


class MixtureSampler:
    def __init__(self, sources: dict[str, tuple[PackedTokens, float]], seq_len: int, batch_size: int,
                 seed: int = 0, rank: int = 0, holdout: float = 0.002) -> None:
        self.names = list(sources)
        self.data = [sources[n][0] for n in self.names]
        w = np.array([sources[n][1] for n in self.names], dtype=np.float64)
        self.weights = w / w.sum()
        self.seq_len, self.batch_size = seq_len, batch_size
        self.seed, self.rank, self.holdout = seed, rank, holdout

    def batch(self, step: int, val: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng([self.seed, self.rank, step, int(val)])
        rows = []
        for _ in range(self.batch_size):
            src = rng.choice(len(self.data), p=self.weights)
            rows.append(self.data[src].sample(rng, self.seq_len + 1, self.holdout, val))
        arr = torch.from_numpy(np.stack(rows))
        return arr[:, :-1].contiguous(), arr[:, 1:].contiguous()


def _pack(seqs: list[tuple[np.ndarray, np.ndarray]], seq_len: int, pad_id: int, pack: bool = True,
          dtype=np.int32) -> tuple[np.ndarray, np.ndarray]:
    """Greedy packing of (ids, loss_mask) pairs into rows of ``seq_len + 1`` tokens.

    Returns compact arrays (ids, mask) of shape (rows, seq_len + 1); labels are
    derived at batch time, so memory is ~(itemsize + 1) bytes per token.
    """
    width = seq_len + 1
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    for i, (ids, _) in enumerate(seqs):
        n = min(len(ids), width)
        if cur and (not pack or cur_len + n > width):
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len += n
    if cur:
        groups.append(cur)
    out_ids = np.full((len(groups), width), pad_id, dtype=dtype)
    out_mask = np.zeros((len(groups), width), dtype=bool)
    for r, g in enumerate(groups):
        pos = 0
        for i in g:
            ids, mask = seqs[i]
            n = min(len(ids), width - pos)
            out_ids[r, pos : pos + n] = ids[:n]
            out_mask[r, pos : pos + n] = mask[:n]
            pos += n
    return out_ids, out_mask


class SFTDataset:
    """Conversations -> packed (input, target) rows; loss only on assistant tokens."""

    def __init__(self, files: list[str], tokenizer: YohoTokenizer, seq_len: int, seed: int = 0,
                 val_fraction: float = 0.01, pack: bool = True) -> None:
        dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.int32
        seqs: list[tuple[np.ndarray, np.ndarray]] = []
        for f in files:
            for rec in iter_records(f):
                enc = tokenizer.encode_chat(rec["messages"])
                if len(enc.ids) < 2 or not any(enc.loss_mask):
                    continue
                seqs.append((np.asarray(enc.ids[: seq_len + 1], dtype=dtype),
                             np.asarray(enc.loss_mask[: seq_len + 1], dtype=bool)))
        random.Random(seed).shuffle(seqs)
        n_val = max(1, int(len(seqs) * val_fraction)) if len(seqs) > 10 else 0
        self.train_ids, self.train_mask = _pack(seqs[n_val:], seq_len, tokenizer.pad_id, pack, dtype)
        self.val_ids, self.val_mask = _pack(seqs[:n_val], seq_len, tokenizer.pad_id, pack, dtype)
        if not len(self.train_ids):
            raise ValueError("SFT dataset is empty")
        self.seq_len = seq_len

    def batch(self, rng: np.random.Generator, batch_size: int, val: bool = False):
        ids, mask = (self.val_ids, self.val_mask) if val and len(self.val_ids) else (self.train_ids, self.train_mask)
        idx = rng.integers(0, len(ids), size=batch_size)
        x = torch.from_numpy(ids[idx].astype(np.int64))
        m = torch.from_numpy(mask[idx])
        labels = torch.where(m, x, torch.full_like(x, -100))
        # labels are aligned with ids; shift for next-token prediction
        return x[:, :-1].contiguous(), labels[:, 1:].contiguous()


class PreferenceDataset:
    def __init__(self, files: list[str], tokenizer: YohoTokenizer, seq_len: int) -> None:
        self.items = []
        for f in files:
            for rec in iter_records(f):
                prompt = tokenizer.encode_chat(rec["prompt"], add_generation_prompt=True).ids
                chosen = tokenizer.encode(rec["chosen"], allow_special=True) + [tokenizer.im_end_id]
                rejected = tokenizer.encode(rec["rejected"], allow_special=True) + [tokenizer.im_end_id]
                self.items.append((prompt, chosen, rejected))
        if not self.items:
            raise ValueError("preference dataset is empty")
        self.seq_len, self.pad = seq_len, tokenizer.pad_id

    def _row(self, prompt: list[int], resp: list[int]) -> tuple[list[int], list[int]]:
        ids = (prompt + resp)[-(self.seq_len + 1):]
        lab = ([-100] * len(prompt) + resp)[-(self.seq_len + 1):]
        pad = self.seq_len + 1 - len(ids)
        return ids + [self.pad] * pad, lab + [-100] * pad

    def batch(self, rng: np.random.Generator, batch_size: int):
        """Returns (ids, labels) of shape (2B, T): first B chosen, last B rejected."""
        idx = rng.integers(0, len(self.items), size=batch_size)
        rows = [self._row(self.items[i][0], self.items[i][1]) for i in idx]
        rows += [self._row(self.items[i][0], self.items[i][2]) for i in idx]
        ids = torch.tensor([r[0] for r in rows])
        lab = torch.tensor([r[1] for r in rows])
        return ids[:, :-1].contiguous(), lab[:, 1:].contiguous()
