"""Run untrusted generated programs with a timeout in an isolated subprocess.

For production-scale evaluation run the whole harness inside a container with
``--network none`` (see deploy/Dockerfile.sandbox); this module adds a process
group, rlimits and a scratch directory but is not a full security boundary.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecOutcome:
    passed: bool
    status: str  # passed | failed | timeout
    output: str


def _limits(mem_mb: int):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))

    return apply


def run_python(program: str, timeout: float = 10.0, mem_mb: int = 2048) -> ExecOutcome:
    with tempfile.TemporaryDirectory(prefix="yoho-exec-") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        proc = subprocess.Popen([sys.executable, "-I", path], cwd=d, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                                preexec_fn=_limits(mem_mb))
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return ExecOutcome(False, "timeout", "")
        text = out.decode("utf-8", "replace")[-4000:]
        return ExecOutcome(proc.returncode == 0, "passed" if proc.returncode == 0 else "failed", text)
