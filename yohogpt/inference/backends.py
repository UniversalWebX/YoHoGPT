"""LLM backends shared by the API server, the agent and the eval harness.

* ``LocalBackend``        - runs a YoHoGPT checkpoint in-process (PyTorch).
* ``OpenAICompatBackend`` - forwards to any OpenAI-compatible server, e.g. a vLLM
                            fleet serving the exported model (production path).
* ``ScriptedBackend``     - returns canned responses (tests / demos without weights).
"""

from __future__ import annotations

import json
from typing import Iterator, Protocol

from ..tokenizer.tokenizer import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from .generate import GenChunk, Generator, SamplingParams


class Backend(Protocol):
    model_name: str

    def chat_stream(self, messages: list[dict], params: SamplingParams) -> Iterator[GenChunk]: ...

    def completion_stream(self, prompt: str, params: SamplingParams, suffix: str | None = None) -> Iterator[GenChunk]: ...


def collect(chunks: Iterator[GenChunk]) -> GenChunk:
    parts, last = [], GenChunk()
    for ch in chunks:
        parts.append(ch.text)
        if ch.finish_reason:
            last = ch
    last.text = "".join(parts)
    return last


class LocalBackend:
    def __init__(self, checkpoint: str, tokenizer: str, device: str | None = None, dtype: str = "auto",
                 model_name: str = "yohogpt") -> None:
        import torch

        from ..tokenizer import YohoTokenizer
        from ..train.checkpoint import load_model

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            tdtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        else:
            tdtype = getattr(torch, dtype)
        self.tok = YohoTokenizer.load(tokenizer)
        self.gen = Generator(load_model(checkpoint, device, tdtype), self.tok, device)
        self.model_name = model_name

    @classmethod
    def from_parts(cls, generator: Generator, model_name: str = "yohogpt") -> "LocalBackend":
        self = cls.__new__(cls)
        self.gen, self.tok, self.model_name = generator, generator.tok, model_name
        return self

    def chat_stream(self, messages, params):
        ids = self.tok.encode_chat(messages, add_generation_prompt=True).ids
        return self.gen.stream(ids, params)

    def completion_stream(self, prompt, params, suffix=None):
        ids = self.tok.encode_fim(prompt, suffix) if suffix is not None else self.tok.encode(prompt)
        return self.gen.stream(ids, params)


class OpenAICompatBackend:
    """Talk to an upstream OpenAI-compatible endpoint (vLLM, TGI, SGLang, ...)."""

    def __init__(self, base_url: str, model_name: str, api_key: str | None = None, timeout: float = 600) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def _payload(self, params: SamplingParams) -> dict:
        p = {"model": self.model_name, "max_tokens": params.max_new_tokens, "temperature": params.temperature,
             "top_p": params.top_p, "stream": True, "stream_options": {"include_usage": True}}
        if params.stop:
            p["stop"] = params.stop
        if params.seed is not None:
            p["seed"] = params.seed
        if params.top_k:
            p["top_k"] = params.top_k
        if params.repetition_penalty != 1.0:
            p["repetition_penalty"] = params.repetition_penalty
        return p

    def _stream(self, path: str, payload: dict, chat: bool) -> Iterator[GenChunk]:
        finish, usage = None, {}
        with self.client.stream("POST", path, json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage") or usage
                for choice in obj.get("choices", []):
                    text = (choice.get("delta") or {}).get("content") if chat else choice.get("text")
                    if text:
                        yield GenChunk(text=text)
                    finish = choice.get("finish_reason") or finish
        yield GenChunk(finish_reason=finish or "stop", prompt_tokens=usage.get("prompt_tokens", 0),
                       completion_tokens=usage.get("completion_tokens", 0))

    def chat_stream(self, messages, params):
        return self._stream("/v1/chat/completions", {**self._payload(params), "messages": messages}, chat=True)

    def completion_stream(self, prompt, params, suffix=None):
        if suffix is not None:
            prompt = f"{FIM_PREFIX}{prompt}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        return self._stream("/v1/completions", {**self._payload(params), "prompt": prompt}, chat=False)


class ScriptedBackend:
    """Deterministic backend: replies with queued responses (or echoes the user)."""

    def __init__(self, responses: list[str] | None = None, model_name: str = "yohogpt-scripted") -> None:
        self.responses = list(responses or [])
        self.model_name = model_name
        self.calls: list[list[dict]] = []

    def _next(self, fallback: str) -> str:
        return self.responses.pop(0) if self.responses else fallback

    def _emit(self, text: str, params: SamplingParams) -> Iterator[GenChunk]:
        for stop in params.stop:
            if stop in text:
                text = text[: text.index(stop)]
        words = text.split(" ")
        for i, w in enumerate(words):
            yield GenChunk(text=w if i == 0 else " " + w)
        yield GenChunk(finish_reason="stop", prompt_tokens=1, completion_tokens=len(words))

    def chat_stream(self, messages, params):
        self.calls.append(messages)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self._emit(self._next(f"You said: {last_user}"), params)

    def completion_stream(self, prompt, params, suffix=None):
        return self._emit(self._next("pass"), params)


def make_backend(kind: str, **kw) -> Backend:
    if kind == "local":
        return LocalBackend(kw["checkpoint"], kw["tokenizer"], kw.get("device"), kw.get("dtype", "auto"),
                            kw.get("model_name", "yohogpt"))
    if kind == "openai":
        return OpenAICompatBackend(kw["upstream"], kw.get("model_name", "yohogpt"), kw.get("upstream_key"))
    if kind == "scripted":
        return ScriptedBackend(kw.get("responses"))
    raise ValueError(f"unknown backend {kind!r}")
