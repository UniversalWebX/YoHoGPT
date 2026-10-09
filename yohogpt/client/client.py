"""Python client + terminal UI for a YoHoGPT server.

    yoho chat  --url http://localhost:8080
    yoho agent --url http://localhost:8080 "Write a CLI that converts CSV to JSON, with tests"
"""

from __future__ import annotations

import json
import os
import sys
from typing import Iterator

import httpx


class YohoClient:
    def __init__(self, url: str = "http://localhost:8080", api_key: str | None = None, timeout: float = 600) -> None:
        api_key = api_key or os.environ.get("YOHO_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=timeout)

    @staticmethod
    def _sse(resp: httpx.Response) -> Iterator[tuple[str, str]]:
        event = "message"
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                yield event, line[5:].strip()
                event = "message"

    def chat_stream(self, messages: list[dict], **params) -> Iterator[str]:
        body = {"model": "yohogpt", "messages": messages, "stream": True, **params}
        with self.http.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            for _, data in self._sse(r):
                if data == "[DONE]":
                    return
                for ch in json.loads(data).get("choices", []):
                    text = ch.get("delta", {}).get("content")
                    if text:
                        yield text

    def chat(self, messages: list[dict], **params) -> str:
        r = self.http.post("/v1/chat/completions", json={"model": "yohogpt", "messages": messages, **params})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def complete(self, prompt: str, suffix: str | None = None, **params) -> str:
        r = self.http.post("/v1/completions", json={"model": "yohogpt", "prompt": prompt, "suffix": suffix, **params})
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def start_agent(self, task: str, **kw) -> dict:
        r = self.http.post("/v1/agent/runs", json={"task": task, **kw})
        r.raise_for_status()
        return r.json()

    def agent_events(self, run_id: str) -> Iterator[dict]:
        with self.http.stream("GET", f"/v1/agent/runs/{run_id}/events") as r:
            r.raise_for_status()
            for event, data in self._sse(r):
                obj = json.loads(data)
                yield {"type": "end", "data": obj} if event == "end" else obj
                if event == "end":
                    return


# ---------------------------------------------------------------- terminal UIs
def chat_repl(url: str, api_key: str | None, system: str | None = None) -> None:
    from ..agent.prompts import SYSTEM_PROMPT_CHAT

    c = YohoClient(url, api_key)
    messages = [{"role": "system", "content": system or SYSTEM_PROMPT_CHAT}]
    print("YoHoGPT chat - /reset to clear, /exit to quit")
    while True:
        try:
            user = input("\n\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user == "/exit":
            return
        if user == "/reset":
            messages = messages[:1]
            continue
        messages.append({"role": "user", "content": user})
        print("\033[1;36myoho>\033[0m ", end="", flush=True)
        parts = []
        for piece in c.chat_stream(messages):
            parts.append(piece)
            print(piece, end="", flush=True)
        print()
        messages.append({"role": "assistant", "content": "".join(parts)})


def agent_cli(url: str, api_key: str | None, task: str, repo_url: str | None = None, max_steps: int = 50) -> int:
    c = YohoClient(url, api_key)
    run = c.start_agent(task, repo_url=repo_url, max_steps=max_steps)
    print(f"started {run['id']}")
    for ev in c.agent_events(run["id"]):
        t, d = ev["type"], ev.get("data", {})
        if t == "assistant":
            thought = d["content"].split("<tool_call>")[0].strip()
            if thought:
                print(f"\033[36m[think]\033[0m {thought}")
        elif t == "tool_call":
            print(f"\033[33m[{d['name']}]\033[0m {json.dumps(d['arguments'])[:300]}")
        elif t == "tool_result":
            out = d["content"]
            print("\033[2m" + (out[:800] + ("..." if len(out) > 800 else "")) + "\033[0m")
        elif t == "error":
            print(f"\033[31m[error]\033[0m {d.get('error')}")
        elif t == "end":
            print(f"\n\033[1m== {d['status']} ==\033[0m\n{d['summary']}")
            if d.get("diff"):
                print("\n" + d["diff"])
            return 0 if d["status"] == "finished" else 1
    return 1


if __name__ == "__main__":  # pragma: no cover
    chat_repl(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080", None)
