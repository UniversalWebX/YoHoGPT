"""``yoho`` command-line entry point.

    yoho sources                       list every training data source + mixture weights
    yoho download    ...               fetch raw data             (yohogpt.data.download)
    yoho train-tokenizer ...           train the BPE tokenizer    (yohogpt.tokenizer.train_tokenizer)
    yoho prepare     ...               filter/dedup/tokenize      (yohogpt.data.prepare)
    yoho plan --gpu NAME --days N      what model size fits your hardware + time budget
    yoho benchmark CONFIG [--find-batch]   measured tokens/s, memory and ETA on this machine
    yoho train CONFIG [k=v ...]        pretrain / SFT / DPO       (yohogpt.train.trainer)
    yoho export-hf   ...               export for vLLM / llama.cpp
    yoho serve       ...               run the API server
    yoho chat  [--url]                 interactive chat against a server
    yoho agent [--url] "task"          run the autonomous coding agent via a server
    yoho run-agent --checkpoint ... "task"   run the agent locally without a server
    yoho eval {humaneval,mbpp} ...     pass@k evaluation
    yoho eval-swebench ...             SWE-bench predictions
"""

from __future__ import annotations

import argparse
import os
import sys


def _serve(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="yoho serve")
    ap.add_argument("--backend", default="local", choices=["local", "openai", "scripted"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--upstream", help="OpenAI-compatible upstream (e.g. vLLM) for --backend openai")
    ap.add_argument("--upstream-key", default=os.environ.get("YOHO_UPSTREAM_KEY"))
    ap.add_argument("--model-name", default="yohogpt")
    ap.add_argument("--device")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sandbox", default=os.environ.get("YOHO_SANDBOX", "docker"), choices=["docker", "local"])
    ap.add_argument("--workers", type=int, default=1, help="uvicorn workers (use 1 with --backend local)")
    args = ap.parse_args(argv)

    import uvicorn

    from .inference.backends import make_backend
    from .server.app import ServerSettings, create_app

    backend = make_backend(args.backend, checkpoint=args.checkpoint, tokenizer=args.tokenizer,
                           upstream=args.upstream, upstream_key=args.upstream_key, model_name=args.model_name,
                           device=args.device)
    settings = ServerSettings(sandbox=args.sandbox)
    uvicorn.run(create_app(backend, settings), host=args.host, port=args.port, proxy_headers=True,
                forwarded_allow_ips="*", timeout_keep_alive=75)


def _run_agent(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="yoho run-agent")
    ap.add_argument("task")
    ap.add_argument("--backend", default="local", choices=["local", "openai"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--upstream")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--sandbox", default="local", choices=["docker", "local"])
    ap.add_argument("--max-steps", type=int, default=50)
    args = ap.parse_args(argv)

    import json

    from .agent import Agent, make_sandbox
    from .inference.backends import make_backend

    backend = make_backend(args.backend, checkpoint=args.checkpoint, tokenizer=args.tokenizer, upstream=args.upstream)

    def show(ev):
        if ev.type == "tool_call":
            print(f"[{ev.step}] {ev.data['name']} {json.dumps(ev.data['arguments'])[:200]}", flush=True)
        elif ev.type == "tool_result":
            print("    " + ev.data["content"][:400].replace("\n", "\n    "), flush=True)

    with make_sandbox(args.sandbox, args.workspace) as sb:
        res = Agent(backend, sb, max_steps=args.max_steps, on_event=show).run(args.task)
    print(f"\n== {res.status} ({res.steps} steps) ==\n{res.summary}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]

    if cmd == "sources":
        from .data.download import main as m

        return m(["--stage", "all", "--list", *rest])
    if cmd == "download":
        from .data.download import main as m

        return m(rest)
    if cmd == "train-tokenizer":
        from .tokenizer.train_tokenizer import main as m

        return m(rest)
    if cmd == "prepare":
        from .data.prepare import main as m

        return m(rest)
    if cmd == "train":
        from .train.trainer import main as m

        return m(rest)
    if cmd == "plan":
        from .plan import main as m

        return m(rest)
    if cmd == "benchmark":
        from .train.benchmark import main as m

        return m(rest)
    if cmd == "export-hf":
        from .export_hf import main as m

        return m(rest)
    if cmd == "serve":
        return _serve(rest)
    if cmd in ("chat", "agent"):
        ap = argparse.ArgumentParser(prog=f"yoho {cmd}")
        ap.add_argument("--url", default=os.environ.get("YOHO_URL", "http://localhost:8080"))
        ap.add_argument("--api-key", default=os.environ.get("YOHO_API_KEY"))
        if cmd == "agent":
            ap.add_argument("task")
            ap.add_argument("--repo-url")
            ap.add_argument("--max-steps", type=int, default=50)
        args = ap.parse_args(rest)
        from .client.client import agent_cli, chat_repl

        if cmd == "chat":
            return chat_repl(args.url, args.api_key)
        sys.exit(agent_cli(args.url, args.api_key, args.task, args.repo_url, args.max_steps))
    if cmd == "run-agent":
        return _run_agent(rest)
    if cmd == "eval":
        from .eval.codegen import main as m

        return m(rest)
    if cmd == "eval-swebench":
        from .eval.swebench import main as m

        return m(rest)
    print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
