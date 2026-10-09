"""Build a small offline corpus (local Python stdlib + this repo) for smoke tests.

    python scripts/make_sample_data.py --out data/raw/pretrain/sample --sft data/sft/sample.jsonl
"""

import argparse
import json
import sysconfig
from pathlib import Path

from yohogpt.data.io import ShardWriter

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/raw/pretrain/sample")
ap.add_argument("--sft", default="data/sft/sample.jsonl")
ap.add_argument("--max-files", type=int, default=400)
args = ap.parse_args()

roots = [Path(sysconfig.get_paths()["stdlib"]), Path(__file__).resolve().parents[1] / "yohogpt"]
n = 0
with ShardWriter(args.out) as w:
    for root in roots:
        for f in sorted(root.rglob("*.py")):
            if "test" in f.parts or "site-packages" in f.parts or n >= args.max_files:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            w.write({"text": text, "meta": {"path": str(f.relative_to(root)), "lang": "python"}})
            n += 1
print(f"wrote {n} files to {args.out}")

Path(args.sft).parent.mkdir(parents=True, exist_ok=True)
with open(args.sft, "w") as fh:
    for a in range(60):
        b = a * 7 % 13
        msgs = [{"role": "user", "content": f"Write a Python function that returns {a} + {b}."},
                {"role": "assistant", "content": f"```python\ndef answer():\n    return {a} + {b}  # {a + b}\n```"}]
        fh.write(json.dumps({"messages": msgs}) + "\n")
print(f"wrote SFT sample to {args.sft}")
