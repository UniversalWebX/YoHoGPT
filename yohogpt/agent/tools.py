"""Tools the YoHoGPT agent can call. Each tool validates its own arguments and
returns plain text that is fed back to the model as a ``tool`` message."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .sandbox import Sandbox

MAX_OUTPUT_CHARS = 12_000


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2 :]
    return f"{head}\n... [{len(text) - limit} characters truncated] ...\n{tail}"


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    fn: Callable[..., str]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def __call__(self, args: dict[str, Any]) -> str:
        props = self.parameters.get("properties", {})
        missing = [k for k in self.parameters.get("required", []) if k not in args]
        if missing:
            raise ToolError(f"missing required arguments: {missing}")
        unknown = [k for k in args if k not in props]
        if unknown:
            raise ToolError(f"unknown arguments: {unknown}")
        return self.fn(**args)


def _obj(required: list[str], **props: dict) -> dict:
    return {"type": "object", "properties": props, "required": required}


def build_tools(sb: Sandbox, default_timeout: int = 120) -> dict[str, Tool]:
    def bash(command: str, timeout: int = default_timeout) -> str:
        r = sb.run(command, timeout=min(int(timeout), 1800))
        return truncate(f"exit_code={r.exit_code}\n{r.output}")

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        p = sb.resolve(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        lines = p.read_text(errors="replace").splitlines()
        end = min(len(lines), end_line or len(lines))
        start = max(1, start_line)
        body = "\n".join(f"{i:>5}\t{lines[i - 1]}" for i in range(start, end + 1))
        return truncate(f"{path} (lines {start}-{end} of {len(lines)})\n{body}")

    def write_file(path: str, content: str) -> str:
        p = sb.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} characters to {path}"

    def str_replace(path: str, old: str, new: str, count: int = 1) -> str:
        p = sb.resolve(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            raise ToolError(f"`old` not found in {path}; read the file and copy the exact text (incl. whitespace)")
        if count != -1 and n != count:
            raise ToolError(f"`old` occurs {n} times in {path}, expected {count}; include more context")
        p.write_text(text.replace(old, new) if count == -1 else text.replace(old, new, count))
        return f"replaced {n if count == -1 else count} occurrence(s) in {path}"

    def list_dir(path: str = ".", depth: int = 2) -> str:
        root = sb.resolve(path)
        if not root.is_dir():
            raise ToolError(f"not a directory: {path}")
        out = []
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "target"}

        def walk(d, level):
            for child in sorted(d.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
                if child.name in skip:
                    continue
                out.append("  " * level + child.name + ("/" if child.is_dir() else ""))
                if child.is_dir() and level + 1 < depth:
                    walk(child, level + 1)
                if len(out) > 500:
                    return

        walk(root, 0)
        return truncate("\n".join(out) or "(empty)")

    def search(pattern: str, path: str = ".", glob: str | None = None) -> str:
        sb.resolve(path)
        cmd = ["grep", "-rnE", "--exclude-dir=.git", "--exclude-dir=node_modules", "-I"]
        if glob:
            cmd.append(f"--include={glob}")
        cmd += ["--", pattern, path]
        r = sb.run(" ".join(_q(c) for c in cmd) + " | head -n 200", timeout=60)
        return truncate(r.output or "no matches")

    def finish(summary: str) -> str:
        return summary

    tools = [
        Tool("bash", "Run a shell command in the workspace (tests, builds, git, package managers).",
             _obj(["command"], command={"type": "string"},
                  timeout={"type": "integer", "description": "seconds, default 120"}), bash),
        Tool("read_file", "Read a text file with line numbers, optionally a line range.",
             _obj(["path"], path={"type": "string"}, start_line={"type": "integer"},
                  end_line={"type": "integer"}), read_file),
        Tool("write_file", "Create or overwrite a file with the given content.",
             _obj(["path", "content"], path={"type": "string"}, content={"type": "string"}), write_file),
        Tool("str_replace", "Replace exact text `old` with `new` in a file. `old` must match exactly "
             "`count` times (default 1; -1 = all).",
             _obj(["path", "old", "new"], path={"type": "string"}, old={"type": "string"},
                  new={"type": "string"}, count={"type": "integer"}), str_replace),
        Tool("list_dir", "List a directory tree.",
             _obj([], path={"type": "string"}, depth={"type": "integer"}), list_dir),
        Tool("search", "Search file contents with an extended regex (grep -E).",
             _obj(["pattern"], pattern={"type": "string"}, path={"type": "string"},
                  glob={"type": "string", "description": "e.g. *.py"}), search),
        Tool("finish", "Finish the task with a summary of the changes and how they were verified.",
             _obj(["summary"], summary={"type": "string"}), finish),
    ]
    return {t.name: t for t in tools}


def _q(s: str) -> str:
    import shlex

    return shlex.quote(s)


def render_tool_specs(tools: dict[str, Tool]) -> str:
    return "\n".join(json.dumps(t.spec()) for t in tools.values())


_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract the first ``<tool_call>{json}</tool_call>`` from model output."""
    m = _CALL_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # tolerate trailing junk after the JSON object
        depth, end = 0, None
        for i, ch in enumerate(raw):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0 and ch == "}":
                end = i + 1
                break
        if end is None:
            raise ToolError("tool call is not valid JSON")
        try:
            obj = json.loads(raw[:end])
        except json.JSONDecodeError as e:
            raise ToolError(f"tool call is not valid JSON: {e}") from e
    if not isinstance(obj, dict) or "name" not in obj:
        raise ToolError('tool call must be {"name": ..., "arguments": {...}}')
    args = obj.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    return obj["name"], args
