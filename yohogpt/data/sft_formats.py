"""Convert heterogeneous instruction / chat / preference datasets into one schema.

Output schemas:
    SFT:  {"messages": [{"role": ..., "content": ...}, ...]}
    DPO:  {"prompt": [messages...], "chosen": str, "rejected": str}
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..agent.prompts import SYSTEM_PROMPT_CHAT

_CHATML_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)
_SHAREGPT_ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                   "system": "system", "tool": "tool", "function": "tool", "observation": "tool"}


def _with_system(messages: list[dict], system: str | None) -> list[dict]:
    if system and (not messages or messages[0]["role"] != "system"):
        return [{"role": "system", "content": system}] + messages
    return messages


def to_messages(record: dict[str, Any], fmt: str, fields: dict[str, str] | None = None,
                text_field: str = "text", system: str | None = SYSTEM_PROMPT_CHAT) -> list[dict] | None:
    """Return a ChatML message list, or None if the record is unusable."""
    fields = fields or {}
    if fmt == "instruction_response":
        inst = record.get(fields.get("instruction", "instruction"))
        resp = record.get(fields.get("response", "response"))
        if not inst or not resp:
            return None
        msgs = [{"role": "user", "content": str(inst).strip()}, {"role": "assistant", "content": str(resp).strip()}]
    elif fmt == "messages":
        raw = record.get(fields.get("messages", "messages"))
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not raw:
            return None
        msgs = []
        for m in raw:
            role = _SHAREGPT_ROLES.get(m.get("role", ""), None)
            if role is None:
                return None
            content = m.get("content") or m.get("text") or ""
            if isinstance(content, list):  # OpenAI content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            msgs.append({"role": role, "content": content})
    elif fmt == "sharegpt":
        conv = record.get(fields.get("conversations", "conversations")) or []
        msgs = []
        for turn in conv:
            role = _SHAREGPT_ROLES.get(turn.get("from", ""))
            if role is None:
                return None
            msgs.append({"role": role, "content": turn.get("value", "")})
    elif fmt == "chatml_text":
        text = record.get(text_field, "")
        msgs = [{"role": r, "content": c.strip()} for r, c in _CHATML_RE.findall(text)]
        msgs = [m for m in msgs if m["role"] in _SHAREGPT_ROLES.values()]
    else:
        raise ValueError(f"unknown SFT format {fmt!r}")

    if not any(m["role"] == "assistant" and m["content"] for m in msgs):
        return None
    return _with_system(msgs, system)


def to_preference(record: dict[str, Any], system: str | None = SYSTEM_PROMPT_CHAT) -> dict | None:
    """Normalise preference records (ultrafeedback-style message lists or flat)."""
    chosen, rejected = record.get("chosen"), record.get("rejected")
    if isinstance(chosen, list) and isinstance(rejected, list):
        if not chosen or chosen[-1].get("role") != "assistant" or not rejected:
            return None
        prompt = chosen[:-1]
        return {"prompt": _with_system(prompt, system), "chosen": chosen[-1]["content"],
                "rejected": rejected[-1]["content"]}
    prompt = record.get("prompt")
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    if not prompt or not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    return {"prompt": _with_system(prompt, system), "chosen": chosen, "rejected": rejected}
