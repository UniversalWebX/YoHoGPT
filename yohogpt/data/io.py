"""Small helpers for reading/writing sharded JSONL(.gz) document files."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


class ShardWriter:
    """Write JSON records into ``shard-00000.jsonl.gz`` files of bounded size."""

    def __init__(self, out_dir: str | Path, max_records: int = 100_000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self.shard = 0
        self.n_in_shard = 0
        self.total = 0
        self._fh = None

    def write(self, record: dict) -> None:
        if self._fh is None or self.n_in_shard >= self.max_records:
            self._roll()
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.n_in_shard += 1
        self.total += 1

    def _roll(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self.shard += 1
        self._fh = gzip.open(self.out_dir / f"shard-{self.shard:05d}.jsonl.gz", "wt", encoding="utf-8")
        self.n_in_shard = 0

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def iter_records(path: str | Path) -> Iterator[dict]:
    """Iterate JSON records from a .jsonl/.jsonl.gz file or a directory of them."""
    path = Path(path)
    files = sorted(path.rglob("*.jsonl*")) if path.is_dir() else [path]
    for f in files:
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
