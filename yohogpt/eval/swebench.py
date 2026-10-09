"""Run the YoHoGPT agent on SWE-bench instances and write a predictions file for
the official harness (https://github.com/SWE-bench/SWE-bench).

    yoho eval-swebench --dataset princeton-nlp/SWE-bench_Verified --upstream http://localhost:8080 \
        --limit 50 --out artifacts/swebench/preds.jsonl
    python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
        --predictions_path artifacts/swebench/preds.jsonl --run_id yoho-v1

The agent edits the repository at ``base_commit``; its ``git diff`` (excluding test
files it may have added) becomes ``model_patch``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ..agent.loop import Agent
from ..agent.sandbox import make_sandbox
from ..inference.backends import make_backend
from ..inference.generate import SamplingParams

TASK_TEMPLATE = """You are working in the repository `{repo}` (checked out in the current directory).
Resolve the following GitHub issue by editing the source code. Do not modify existing tests.
Reproduce the problem first if you can, then fix it and verify the fix.

<issue>
{problem}
</issue>"""


def prepare_repo(instance: dict, root: Path) -> Path:
    ws = root / instance["instance_id"]
    if ws.exists():
        shutil.rmtree(ws)
    url = f"https://github.com/{instance['repo']}.git"
    subprocess.run(["git", "clone", "--quiet", url, str(ws)], check=True)
    subprocess.run(["git", "checkout", "--quiet", instance["base_commit"]], cwd=ws, check=True)
    return ws


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--instances", nargs="*", help="only these instance ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--backend", default="openai", choices=["local", "openai"])
    ap.add_argument("--upstream", default="http://localhost:8080")
    ap.add_argument("--upstream-key")
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer")
    ap.add_argument("--sandbox", default="docker", choices=["docker", "local"])
    ap.add_argument("--sandbox-image", default="yohogpt/sandbox:latest")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--workspaces", default="workspaces/swebench")
    ap.add_argument("--out", default="artifacts/swebench/preds.jsonl")
    ap.add_argument("--model-name", default="yohogpt")
    args = ap.parse_args(argv)

    from datasets import load_dataset

    ds = [dict(r) for r in load_dataset(args.dataset, split=args.split)]
    if args.instances:
        ds = [r for r in ds if r["instance_id"] in set(args.instances)]
    ds = ds[: args.limit]
    backend = make_backend(args.backend, upstream=args.upstream, upstream_key=args.upstream_key,
                           checkpoint=args.checkpoint, tokenizer=args.tokenizer)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["instance_id"] for line in out.read_text().splitlines()} if out.exists() else set()

    for inst in ds:
        if inst["instance_id"] in done:
            continue
        ws = prepare_repo(inst, Path(args.workspaces))
        kw = {"image": args.sandbox_image} if args.sandbox == "docker" else {}
        with make_sandbox(args.sandbox, ws, **kw) as sb:
            agent = Agent(backend, sb, max_steps=args.max_steps,
                          params=SamplingParams(max_new_tokens=2048, temperature=0.0),
                          on_event=lambda e: print(f"[{inst['instance_id']}] step {e.step}: {e.type}"))
            result = agent.run(TASK_TEMPLATE.format(repo=inst["repo"], problem=inst["problem_statement"]))
        with open(out, "a") as f:
            f.write(json.dumps({"instance_id": inst["instance_id"], "model_name_or_path": args.model_name,
                                "model_patch": result.diff, "status": result.status}) + "\n")
        print(f"[{inst['instance_id']}] {result.status} after {result.steps} steps, patch={len(result.diff)} chars")


if __name__ == "__main__":
    main()
