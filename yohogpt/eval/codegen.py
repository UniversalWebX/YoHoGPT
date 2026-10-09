"""Functional-correctness evaluation (HumanEval / MBPP style) with pass@k.

    yoho eval humaneval --backend openai --upstream http://localhost:8000 -n 10 -k 1 10
    yoho eval mbpp --problems data/eval/mbpp.jsonl --mode chat --emit-pairs data/sft/dpo_pairs.jsonl

Problem JSONL schema (HumanEval): task_id, prompt, test, entry_point
                      (MBPP):     task_id, text, code, test_list
The same harness produces execution-verified training data:
  --emit-sft   passing chat samples   -> rejection-sampling SFT data
  --emit-pairs (pass, fail) per prompt -> DPO preference pairs
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..agent.prompts import SYSTEM_PROMPT_CHAT
from ..data.io import iter_records
from ..inference.backends import Backend, collect
from ..inference.generate import SamplingParams
from .execution import run_python

HF_IDS = {"humaneval": ("openai/openai_humaneval", "test"), "mbpp": ("google-research-datasets/mbpp", "test")}
_CODE_BLOCK = re.compile(r"```(?:python|py)?\n(.*?)```", re.S)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from Chen et al. (2021)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def load_problems(name: str, path: str | None) -> list[dict]:
    if path:
        return list(iter_records(path))
    from datasets import load_dataset

    hf_id, split = HF_IDS[name]
    return [dict(r) for r in load_dataset(hf_id, split=split)]


def normalise_problem(p: dict) -> dict:
    """Map MBPP-style problems onto the HumanEval schema."""
    if "prompt" in p and "entry_point" in p:
        return p
    fn = re.search(r"def\s+(\w+)\s*\(", p["code"]).group(1)
    tests = "\n".join(p["test_list"])
    return {"task_id": str(p["task_id"]), "prompt": f'"""{p["text"]}\n{p["test_list"][0]}\n"""\n',
            "test": f"def check(candidate):\n    pass\n\n{p.get('test_setup_code', '')}\n{tests}\n",
            "entry_point": fn, "instruction": f"{p['text']}\nYour code should pass this test:\n{p['test_list'][0]}"}


def chat_prompt(p: dict) -> list[dict]:
    inst = p.get("instruction") or (
        "Complete the following Python function. Return the full function in a ```python code block.\n\n"
        f"```python\n{p['prompt']}```")
    return [{"role": "system", "content": SYSTEM_PROMPT_CHAT}, {"role": "user", "content": inst}]


def build_program(p: dict, sample: str, mode: str) -> str:
    if mode == "chat":
        m = _CODE_BLOCK.search(sample)
        code = m.group(1) if m else sample
        body = p["prompt"] + "\n" + code
    else:
        body = p["prompt"] + sample
    return f"{body}\n\n{p['test']}\n\ncheck({p['entry_point']})\n"


def evaluate(backend: Backend, problems: list[dict], n: int, ks: list[int], mode: str, temperature: float,
             max_tokens: int, workers: int = 8) -> tuple[dict, list[dict]]:
    stops = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#"] if mode == "base" else []
    params = SamplingParams(max_new_tokens=max_tokens, temperature=temperature if n > 1 else 0.0, top_p=0.95,
                            stop=stops)

    def one(job):
        p, i = job
        if mode == "chat":
            text = collect(backend.chat_stream(chat_prompt(p), params)).text
        else:
            text = collect(backend.completion_stream(p["prompt"], params)).text
        outcome = run_python(build_program(p, text, mode))
        return {"task_id": p["task_id"], "sample": i, "completion": text, "passed": outcome.passed,
                "status": outcome.status}

    jobs = [(p, i) for p in problems for i in range(n)]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, jobs))

    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r["passed"])
    scores = {f"pass@{k}": float(np.mean([pass_at_k(len(v), sum(v), k) for v in by_task.values()]))
              for k in ks if k <= n}
    return scores, results


def emit_training_data(problems: list[dict], results: list[dict], sft_path: str | None, pairs_path: str | None) -> None:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)
    probs = {p["task_id"]: p for p in problems}
    if sft_path:
        with open(sft_path, "a") as f:
            for tid, rs in by_task.items():
                for r in rs:
                    if r["passed"]:
                        msgs = chat_prompt(probs[tid]) + [{"role": "assistant", "content": r["completion"]}]
                        f.write(json.dumps({"messages": msgs}) + "\n")
                        break
    if pairs_path:
        with open(pairs_path, "a") as f:
            for tid, rs in by_task.items():
                good = [r for r in rs if r["passed"]]
                bad = [r for r in rs if not r["passed"]]
                for g, b in zip(good, bad):
                    f.write(json.dumps({"prompt": chat_prompt(probs[tid]), "chosen": g["completion"],
                                        "rejected": b["completion"]}) + "\n")


def main(argv: list[str] | None = None) -> None:
    from ..inference.backends import make_backend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["humaneval", "mbpp"])
    ap.add_argument("--problems", help="local JSONL (otherwise downloaded from the HF Hub)")
    ap.add_argument("--backend", default="openai", choices=["local", "openai"])
    ap.add_argument("--upstream", default="http://localhost:8080")
    ap.add_argument("--upstream-key")
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer")
    ap.add_argument("--mode", default="chat", choices=["chat", "base"])
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("-k", type=int, nargs="+", default=[1])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--emit-sft")
    ap.add_argument("--emit-pairs")
    args = ap.parse_args(argv)

    backend = make_backend(args.backend, upstream=args.upstream, upstream_key=args.upstream_key,
                           checkpoint=args.checkpoint, tokenizer=args.tokenizer)
    problems = [normalise_problem(p) for p in load_problems(args.benchmark, args.problems)][: args.limit]
    scores, results = evaluate(backend, problems, args.n, args.k, args.mode, args.temperature, args.max_tokens,
                               args.workers if args.backend == "openai" else 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.benchmark}_samples.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    (out / f"{args.benchmark}_scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps({"benchmark": args.benchmark, "problems": len(problems), **scores}, indent=2))
    emit_training_data(problems, results, args.emit_sft, args.emit_pairs)


if __name__ == "__main__":
    main()
