"""Autoregressive generation with KV cache, nucleus sampling and stop strings."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch

from ..model import YohoGPT
from ..tokenizer import YohoTokenizer
from ..tokenizer.tokenizer import StreamDecoder


@dataclass
class SamplingParams:
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass
class GenChunk:
    text: str = ""
    finish_reason: str | None = None  # "stop" | "length" | None while streaming
    prompt_tokens: int = 0
    completion_tokens: int = 0


def sample_next(logits: torch.Tensor, params: SamplingParams, history: torch.Tensor | None,
                generator: torch.Generator | None) -> int:
    logits = logits.float().clone()
    if params.repetition_penalty != 1.0 and history is not None and history.numel():
        prev = logits[history]
        logits[history] = torch.where(prev > 0, prev / params.repetition_penalty, prev * params.repetition_penalty)
    if params.temperature <= 0:
        return int(logits.argmax())
    logits /= params.temperature
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if 0 < params.top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = sp.cumsum(0)
        sp[(cum - sp) > params.top_p] = 0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs /= probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


class Generator:
    """Thread-safe single-model generator (one sequence at a time per model).

    For high-throughput serving export to HF format and run vLLM behind the
    gateway (see ``yohogpt.server.backends.OpenAICompatBackend``).
    """

    def __init__(self, model: YohoGPT, tokenizer: YohoTokenizer, device: torch.device | str = "cpu") -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self.device = torch.device(device)
        self.lock = threading.Lock()
        self.stop_ids = {tokenizer.eos_id, tokenizer.im_end_id}

    @torch.inference_mode()
    def stream(self, prompt_ids: list[int], params: SamplingParams) -> Iterator[GenChunk]:
        max_len = self.model.cfg.max_seq_len
        budget = max(1, min(params.max_new_tokens, max_len - 1))
        if len(prompt_ids) + budget > max_len:  # keep the most recent context
            prompt_ids = prompt_ids[-(max_len - budget):]
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(params.seed)

        with self.lock:
            self.model.setup_cache(1, len(prompt_ids) + budget)
            try:
                yield from self._decode(prompt_ids, params, budget, gen)
            finally:
                self.model.clear_cache()

    def _decode(self, prompt_ids, params, budget, gen) -> Iterator[GenChunk]:
        idx = torch.tensor([prompt_ids], device=self.device)
        logits, _ = self.model(idx, use_cache=True, last_only=True)
        pos = len(prompt_ids)
        decoder = StreamDecoder(self.tok)
        history: list[int] = []
        text_out = ""
        held = ""  # text withheld because it may be the start of a stop string
        max_stop = max((len(s) for s in params.stop), default=0)
        finish = "length"
        hit_stop_string = False

        for _ in range(budget):
            nxt = sample_next(logits[0, -1].cpu(), params,
                              torch.tensor(history, dtype=torch.long) if history else None, gen)
            if nxt in self.stop_ids:
                finish = "stop"
                break
            history.append(nxt)
            delta = decoder.push(nxt)
            if delta:
                held += delta
                hit = _find_stop(text_out + held, params.stop, len(text_out))
                if hit is not None:
                    final = (text_out + held)[:hit]
                    if len(final) > len(text_out):
                        yield GenChunk(text=final[len(text_out):])
                    finish, hit_stop_string = "stop", True
                    break
                safe = len(held) - (max_stop - 1) if max_stop else len(held)
                if safe > 0:
                    yield GenChunk(text=held[:safe])
                    text_out += held[:safe]
                    held = held[safe:]
            if pos >= self.model.cfg.max_seq_len:
                break
            logits, _ = self.model(torch.tensor([[nxt]], device=self.device), start_pos=pos, use_cache=True)
            pos += 1

        if held and not hit_stop_string:
            yield GenChunk(text=held)
        yield GenChunk(finish_reason=finish, prompt_tokens=len(prompt_ids), completion_tokens=len(history))

    def complete(self, prompt_ids: list[int], params: SamplingParams) -> GenChunk:
        out = GenChunk()
        start = time.time()
        parts = []
        for ch in self.stream(prompt_ids, params):
            parts.append(ch.text)
            if ch.finish_reason:
                out = ch
        out.text = "".join(parts)
        out.__dict__["latency_s"] = time.time() - start
        return out


def _find_stop(text: str, stops: list[str], search_from: int) -> int | None:
    best = None
    for s in stops:
        i = text.find(s, max(0, search_from - len(s)))
        if i != -1 and (best is None or i < best):
            best = i
    return best
