"""Execution sandboxes for the agent.

* ``DockerSandbox`` (production): a dedicated container per session, no network by
  default, CPU/memory/pids limits, workspace bind-mounted at /workspace.
* ``LocalSandbox`` (development / tests): subprocesses in a workspace directory with
  timeouts and rlimits. NOT a security boundary - only use with trusted tasks.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


class Sandbox:
    workspace: Path

    def run(self, command: str, timeout: int = 120) -> ExecResult:  # pragma: no cover - interface
        raise NotImplementedError

    def resolve(self, rel: str) -> Path:
        """Resolve a workspace-relative path, refusing escapes outside the workspace."""
        p = (self.workspace / rel).resolve()
        root = self.workspace.resolve()
        if p != root and root not in p.parents:
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _limit_resources(mem_mb: int, cpu_s: int):
    def apply() -> None:  # runs in the child before exec
        import resource

        os.setsid()
        if mem_mb > 0:  # address-space cap; JVM/Node/Go reserve lots of VA, so off by default
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))

    return apply


class LocalSandbox(Sandbox):
    def __init__(self, workspace: str | Path, mem_mb: int = 0, env: dict[str, str] | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mem_mb = mem_mb
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.workspace),
                    "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}

    def run(self, command: str, timeout: int = 120) -> ExecResult:
        proc = subprocess.Popen(
            ["bash", "-lc", command], cwd=self.workspace, env=self.env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, preexec_fn=_limit_resources(self.mem_mb, timeout + 5),
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            return ExecResult(proc.returncode, out.decode("utf-8", "replace"))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return ExecResult(-9, out.decode("utf-8", "replace") + f"\n[timed out after {timeout}s]", True)


class DockerSandbox(Sandbox):
    def __init__(self, workspace: str | Path, image: str = "yohogpt/sandbox:latest", network: bool = False,
                 cpus: float = 2.0, mem: str = "4g", pids: int = 512) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("docker is not installed")
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.name = f"yoho-sbx-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker", "run", "-d", "--rm", "--name", self.name,
            "--cpus", str(cpus), "--memory", mem, "--pids-limit", str(pids),
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "-v", f"{self.workspace}:/workspace", "-w", "/workspace",
        ]
        if not network:
            cmd += ["--network", "none"]
        cmd += [image, "sleep", "infinity"]
        subprocess.run(cmd, check=True, capture_output=True)

    def run(self, command: str, timeout: int = 120) -> ExecResult:
        wrapped = f"timeout -s KILL {timeout} bash -lc {shlex.quote(command)}"
        try:
            r = subprocess.run(["docker", "exec", self.name, "bash", "-c", wrapped], capture_output=True,
                               timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            return ExecResult(-9, f"[timed out after {timeout}s]", True)
        out = (r.stdout + r.stderr).decode("utf-8", "replace")
        timed_out = r.returncode == 137
        if timed_out:
            out += f"\n[timed out after {timeout}s]"
        return ExecResult(r.returncode, out, timed_out)

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def make_sandbox(kind: str, workspace: str | Path, **kw) -> Sandbox:
    if kind == "docker":
        return DockerSandbox(workspace, **kw)
    if kind == "local":
        return LocalSandbox(workspace, **kw)
    raise ValueError(f"unknown sandbox kind {kind!r}")
