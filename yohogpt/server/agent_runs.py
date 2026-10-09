"""Server-side management of agent runs (each run = one sandboxed workspace + thread)."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.loop import Agent, AgentEvent
from ..agent.sandbox import make_sandbox
from ..inference.backends import Backend
from ..inference.generate import SamplingParams
from .schemas import AgentRunRequest

_SAFE_GIT_URL = re.compile(r"^(https://|git@)[\w.\-]+[:/][\w.\-/]+?(\.git)?/?$")


@dataclass
class AgentRun:
    id: str
    owner: str
    task: str
    workspace: Path
    status: str = "queued"  # queued | running | finished | max_steps | cancelled | error
    events: list[dict] = field(default_factory=list)
    summary: str = ""
    diff: str = ""
    created: float = field(default_factory=time.time)
    agent: Agent | None = None
    cond: threading.Condition = field(default_factory=threading.Condition)

    def add_event(self, ev: AgentEvent) -> None:
        with self.cond:
            self.events.append(ev.to_dict())
            self.cond.notify_all()

    def wait_events(self, after: int, timeout: float) -> list[dict]:
        with self.cond:
            if len(self.events) <= after and not self.done:
                self.cond.wait(timeout)
            return self.events[after:]

    @property
    def done(self) -> bool:
        return self.status not in ("queued", "running")

    def public(self, include_events: bool = False) -> dict:
        d = {"id": self.id, "status": self.status, "task": self.task, "summary": self.summary,
             "diff": self.diff, "created": self.created, "num_events": len(self.events)}
        if include_events:
            d["events"] = self.events
        return d


class AgentRunManager:
    def __init__(self, backend: Backend, workspaces: str | Path, sandbox: str = "docker",
                 max_concurrent: int = 4, sandbox_image: str = "yohogpt/sandbox:latest") -> None:
        self.backend = backend
        self.root = Path(workspaces)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sandbox_kind = sandbox
        self.sandbox_image = sandbox_image
        self.pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="agent")
        self.runs: dict[str, AgentRun] = {}
        self.lock = threading.Lock()

    def create(self, req: AgentRunRequest, owner: str) -> AgentRun:
        if req.repo_url and not _SAFE_GIT_URL.match(req.repo_url):
            raise ValueError("repo_url must be an https:// or git@ URL")
        run_id = "run_" + uuid.uuid4().hex[:16]
        ws = self.root / run_id
        ws.mkdir(parents=True)
        for rel, content in (req.files or {}).items():
            p = (ws / rel).resolve()
            if ws.resolve() not in p.parents:
                raise ValueError(f"file path escapes workspace: {rel}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        run = AgentRun(run_id, owner, req.task, ws)
        with self.lock:
            self.runs[run_id] = run
        self.pool.submit(self._execute, run, req)
        return run

    def get(self, run_id: str, owner: str) -> AgentRun | None:
        run = self.runs.get(run_id)
        return run if run is not None and run.owner == owner else None

    def list(self, owner: str) -> list[AgentRun]:
        return [r for r in self.runs.values() if r.owner == owner]

    def _execute(self, run: AgentRun, req: AgentRunRequest) -> None:
        run.status = "running"
        try:
            if req.repo_url:
                cmd = ["git", "clone", "--depth", "50"]
                if req.branch:
                    cmd += ["--branch", req.branch]
                subprocess.run(cmd + ["--", req.repo_url, str(run.workspace / "repo_tmp")], check=True,
                               capture_output=True, timeout=600)
                for item in (run.workspace / "repo_tmp").iterdir():
                    shutil.move(str(item), run.workspace / item.name)
                (run.workspace / "repo_tmp").rmdir()
            kw = {"image": self.sandbox_image} if self.sandbox_kind == "docker" else {}
            with make_sandbox(self.sandbox_kind, run.workspace, **kw) as sb:
                agent = Agent(self.backend, sb, max_steps=req.max_steps, on_event=run.add_event,
                              params=SamplingParams(max_new_tokens=2048, temperature=req.temperature))
                run.agent = agent
                result = agent.run(req.task)
            run.summary, run.diff = result.summary, result.diff
            final = result.status
        except Exception as e:  # noqa: BLE001 - surface any failure to the client
            run.summary = f"run failed: {e}"
            run.add_event(AgentEvent("error", 0, {"error": str(e)}))
            final = "error"
        with run.cond:
            run.status = final
            run.cond.notify_all()

    def shutdown(self) -> None:
        for run in self.runs.values():
            if run.agent:
                run.agent.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
