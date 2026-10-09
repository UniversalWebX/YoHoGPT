"""The YoHoGPT agent loop (Devin-style: plan -> act -> observe -> repeat).

    agent = Agent(backend, LocalSandbox("workspaces/demo"))
    result = agent.run("Add a --verbose flag to cli.py and test it")
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from ..inference.backends import Backend, collect
from ..inference.generate import SamplingParams
from ..tokenizer.tokenizer import TOOL_CALL, TOOL_CALL_END
from .prompts import SYSTEM_PROMPT_AGENT
from .sandbox import Sandbox
from .tools import ToolError, build_tools, parse_tool_call, render_tool_specs


@dataclass
class AgentEvent:
    type: str  # assistant | tool_call | tool_result | user | error | done
    step: int
    data: dict
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "step": self.step, "data": self.data, "ts": self.ts}


@dataclass
class AgentResult:
    status: str  # finished | max_steps | cancelled | error
    summary: str
    steps: int
    messages: list[dict]
    diff: str = ""


class Agent:
    def __init__(
        self,
        backend: Backend,
        sandbox: Sandbox,
        max_steps: int = 50,
        params: SamplingParams | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        context_chars: int = 200_000,
        keep_recent_results: int = 6,
    ) -> None:
        self.backend = backend
        self.sandbox = sandbox
        self.tools = build_tools(sandbox)
        self.max_steps = max_steps
        self.params = params or SamplingParams(max_new_tokens=2048, temperature=0.2, top_p=0.95)
        self.params.stop = list({*self.params.stop, TOOL_CALL_END})
        self.on_event = on_event or (lambda e: None)
        self.context_chars = context_chars
        self.keep_recent_results = keep_recent_results
        self.inbox: "queue.Queue[str]" = queue.Queue()
        self.cancelled = threading.Event()
        self.messages: list[dict] = []

    # ----------------------------------------------------------- control
    def send_user_message(self, text: str) -> None:
        """Inject guidance mid-run (delivered before the next model call)."""
        self.inbox.put(text)

    def cancel(self) -> None:
        self.cancelled.set()

    # -------------------------------------------------------------- main
    def run(self, task: str) -> AgentResult:
        system = SYSTEM_PROMPT_AGENT.replace("{tools}", render_tool_specs(self.tools))
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        self._emit("user", 0, {"content": task})
        no_tool_streak = 0

        for step in range(1, self.max_steps + 1):
            if self.cancelled.is_set():
                return self._result("cancelled", "Run cancelled by user.", step - 1)
            while not self.inbox.empty():
                msg = self.inbox.get_nowait()
                self.messages.append({"role": "user", "content": msg})
                self._emit("user", step, {"content": msg})

            self._compact()
            try:
                text = collect(self.backend.chat_stream(self.messages, self.params)).text
            except Exception as e:  # backend/network failure
                self._emit("error", step, {"error": f"model backend error: {e}"})
                return self._result("error", f"model backend error: {e}", step)
            if TOOL_CALL in text and TOOL_CALL_END not in text:
                text += TOOL_CALL_END  # the stop sequence is stripped by the backend
            self.messages.append({"role": "assistant", "content": text})
            self._emit("assistant", step, {"content": text})

            try:
                call = parse_tool_call(text)
            except ToolError as e:
                self._tool_message(step, f"error: {e}")
                continue

            if call is None:
                no_tool_streak += 1
                if no_tool_streak >= 2:  # model is just talking; treat the reply as the answer
                    return self._result("finished", text.strip(), step)
                self.messages.append({"role": "user", "content": "Continue with a tool call, or call `finish` "
                                      "with a summary if the task is complete."})
                continue
            no_tool_streak = 0

            name, args = call
            self._emit("tool_call", step, {"name": name, "arguments": args})
            tool = self.tools.get(name)
            if tool is None:
                self._tool_message(step, f"error: unknown tool {name!r}. Available: {', '.join(self.tools)}")
                continue
            try:
                output = tool(args)
            except (ToolError, PermissionError, TypeError, ValueError, OSError) as e:
                self._tool_message(step, f"error: {e}")
                continue
            if name == "finish":
                return self._result("finished", output, step)
            self._tool_message(step, output)

        return self._result("max_steps", "Reached the step limit before finishing.", self.max_steps)

    # ----------------------------------------------------------- helpers
    def _emit(self, type_: str, step: int, data: dict) -> None:
        self.on_event(AgentEvent(type_, step, data))

    def _tool_message(self, step: int, content: str) -> None:
        self.messages.append({"role": "tool", "content": content})
        self._emit("tool_result", step, {"content": content})

    def _compact(self) -> None:
        """Elide old tool outputs when the transcript outgrows the context budget."""
        total = sum(len(m["content"]) for m in self.messages)
        if total <= self.context_chars:
            return
        tool_idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        for i in tool_idx[: -self.keep_recent_results]:
            if total <= self.context_chars:
                break
            old = self.messages[i]["content"]
            if old.startswith("[elided"):
                continue
            stub = f"[elided {len(old)} chars of earlier tool output to save context]"
            self.messages[i] = {"role": "tool", "content": stub}
            total -= len(old) - len(stub)

    def _result(self, status: str, summary: str, steps: int) -> AgentResult:
        diff = ""
        if (self.sandbox.workspace / ".git").exists():
            diff = self.sandbox.run("git add -A -N . && git diff", timeout=60).output
        res = AgentResult(status, summary, steps, self.messages, diff)
        self._emit("done", steps, {"status": status, "summary": summary, "diff": diff})
        return res
