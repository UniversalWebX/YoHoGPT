"""Download every source in ``data/sources.yaml`` into normalised JSONL shards.

    yoho download --stage pretrain --out data/raw [--only starcoderdata] [--limit 10000]

Output layout:  data/raw/<stage>/<source>/shard-00000.jsonl.gz
Pretrain records:  {"text": str, "meta": {...}}
SFT records:       {"messages": [...], "meta": {...}}
DPO records:       {"prompt": [...], "chosen": str, "rejected": str, "meta": {...}}
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

from .filters import detect_license, language_of, redact
from .io import ShardWriter, iter_records
from .sft_formats import to_messages, to_preference
from .sources import Catalogue, Source, load_catalogue

SKIP_DIRS = {".git", "node_modules", "vendor", "third_party", "build", "dist", "__pycache__", ".venv"}


# ------------------------------------------------------------------ HF datasets
def iter_hf(src: Source, limit: int | None) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install 'yohogpt[data]' to download HuggingFace datasets") from e
    ds = load_dataset(src.hf_id, name=src.name_config, data_dir=src.data_dir, split=src.split, streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield row


# ------------------------------------------------------------------ git repos
def repo_license(repo_dir: Path) -> str | None:
    for pattern in ("LICENSE*", "LICENCE*", "COPYING*", "license*"):
        for f in sorted(repo_dir.glob(pattern)):
            if f.is_file():
                lic = detect_license(f.read_text(errors="replace")[:20_000])
                if lic:
                    return lic
    return None


def iter_git(src: Source, cache: Path, allowlist: list[str], limit: int | None) -> Iterator[dict]:
    n = 0
    for url in src.repos:
        name = url.rstrip("/").split("/")[-2] + "__" + url.rstrip("/").split("/")[-1]
        dest = cache / name
        if not dest.exists():
            print(f"[git] cloning {url}", file=sys.stderr)
            r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(dest)])
            if r.returncode != 0:
                print(f"[git] clone failed: {url}", file=sys.stderr)
                continue
        lic = repo_license(dest)
        if lic is None or not any(lic.startswith(a) for a in allowlist):
            print(f"[git] skipping {url}: license {lic!r} not in allowlist", file=sys.stderr)
            continue
        for f in dest.rglob("*"):
            if not f.is_file() or any(part in SKIP_DIRS for part in f.relative_to(dest).parts):
                continue
            rel = f.relative_to(dest).as_posix()
            if src.include_globs and not any(fnmatch.fnmatch(rel, g) for g in src.include_globs):
                continue
            lang = language_of(rel)
            if lang is None or f.stat().st_size > 1_000_000:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            yield {"text": text, "meta": {"repo": url, "path": rel, "lang": lang, "license": lic}}
            n += 1
            if limit is not None and n >= limit:
                return


# ------------------------------------------------------------------ direct URLs
def iter_url(src: Source, cache: Path, limit: int | None) -> Iterator[dict]:
    assert src.url
    if not src.url.endswith((".jsonl", ".jsonl.gz", ".json")):
        print(
            f"[url] {src.name}: {src.url} is not a direct JSONL file. Download/extract it manually "
            f"(see notes in sources.yaml), convert to JSONL with a 'text' field and place it under "
            f"{cache / src.name}. For Stack Exchange dumps use `python -m yohogpt.data.stackexchange`.",
            file=sys.stderr,
        )
        local = cache / src.name
        if local.exists():
            yield from _limited(iter_records(local), limit)
        return
    dest = cache / Path(src.url).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(src.url) as resp, open(dest, "wb") as out:  # noqa: S310 - catalogue URL
            shutil.copyfileobj(resp, out)
    yield from _limited(iter_records(dest), limit)


def _limited(it: Iterator[dict], limit: int | None) -> Iterator[dict]:
    for i, r in enumerate(it):
        if limit is not None and i >= limit:
            return
        yield r


# ------------------------------------------------------------------ driver
def records_for(src: Source, cat: Catalogue, cache: Path, limit: int | None) -> Iterator[dict]:
    if src.kind == "hf":
        return iter_hf(src, limit)
    if src.kind == "git":
        return iter_git(src, cache / "_git", cat.license_allowlist, limit)
    if src.kind == "url":
        return iter_url(src, cache / "_url", limit)
    if src.kind == "local":
        p = Path(src.path or "")
        return _limited(iter_records(p), limit) if p.exists() else iter(())
    raise ValueError(f"unknown source kind {src.kind}")


def normalise(src: Source, row: dict) -> dict | None:
    meta = {"source": src.name, "license": src.license}
    if src.stage == "sft":
        msgs = row.get("messages") if src.kind == "local" else None
        msgs = msgs or to_messages(row, src.format or "messages", src.fields, src.text_field)
        return {"messages": msgs, "meta": meta} if msgs else None
    if src.stage == "dpo":
        pref = to_preference(row)
        return {**pref, "meta": meta} if pref else None
    text = row.get(src.text_field) if src.kind == "hf" else row.get("text")
    if not text or not isinstance(text, str):
        return None
    meta.update(row.get("meta", {}))
    for k in ("max_stars_repo_path", "path", "lang", "language"):
        if k in row and isinstance(row[k], str):
            meta[k] = row[k]
    return {"text": redact(text), "meta": meta}


def download_source(src: Source, cat: Catalogue, out: Path, limit: int | None) -> int:
    dest = out / src.stage / src.name
    written = 0
    with ShardWriter(dest) as w:
        for row in records_for(src, cat, out, limit):
            rec = normalise(src, row)
            if rec is not None:
                w.write(rec)
                written += 1
    print(f"[{src.stage}/{src.name}] wrote {written} records -> {dest}", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--stage", default="pretrain", help="pretrain | long_context | sft | dpo | all")
    ap.add_argument("--only", nargs="*", help="source names to download")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit", type=int, default=None, help="max records per source (for smoke tests)")
    ap.add_argument("--list", action="store_true", help="print the catalogue and exit")
    args = ap.parse_args(argv)

    cat = load_catalogue(args.catalogue) if args.catalogue else load_catalogue()
    stages = ["pretrain", "long_context", "sft", "dpo"] if args.stage == "all" else [args.stage]
    if args.list:
        for st in stages:
            print(f"== {st}")
            for name, w in cat.mixture_weights(st).items():
                s = cat.get(name, st)
                print(f"  {name:28s} {w:6.1%}  {s.kind:5s} {s.hf_id or s.url or s.path or ''}  [{s.license}]")
        return
    for st in stages:
        for src in cat.stage(st):
            if args.only and src.name not in args.only:
                continue
            download_source(src, cat, Path(args.out), args.limit)


if __name__ == "__main__":
    main()
