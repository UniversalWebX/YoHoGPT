"""YoHoGPT tokenizer: byte-level BPE + chat / FIM / tool-call special tokens.

Chat format (ChatML):

    <|im_start|>system
    You are YoHoGPT...<|im_end|>
    <|im_start|>user
    Fix the bug in utils.py<|im_end|>
    <|im_start|>assistant
    <tool_call>{"name": "read_file", "arguments": {"path": "utils.py"}}</tool_call><|im_end|>
    <|im_start|>tool
    ...file contents...<|im_end|>
    <|im_start|>assistant

Fill-in-the-middle (for IDE completion) uses PSM order:

    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer

EOS = "<|endoftext|>"
PAD = "<|pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
FILE_SEP = "<|file_sep|>"
REPO_NAME = "<|repo_name|>"
TOOL_CALL = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
THINK = "<think>"
THINK_END = "</think>"

# Order matters: ids 0..N-1 are assigned in this order by the trainer.
SPECIAL_TOKENS: list[str] = [
    EOS,
    PAD,
    IM_START,
    IM_END,
    FIM_PREFIX,
    FIM_MIDDLE,
    FIM_SUFFIX,
    FILE_SEP,
    REPO_NAME,
    TOOL_CALL,
    TOOL_CALL_END,
    THINK,
    THINK_END,
]

ROLES = ("system", "user", "assistant", "tool")


@dataclass
class EncodedChat:
    ids: list[int]
    loss_mask: list[int]  # 1 where the model should learn to produce the token


class YohoTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tk = tokenizer
        self.special = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
        missing = [t for t, i in self.special.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer is missing special tokens: {missing}")
        self.eos_id = self.special[EOS]
        self.pad_id = self.special[PAD]
        self.im_start_id = self.special[IM_START]
        self.im_end_id = self.special[IM_END]

    @classmethod
    def load(cls, path: str | Path) -> "YohoTokenizer":
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self.tk.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    # ----------------------------------------------------------- basic I/O
    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text. Special-token strings in untrusted text are NOT treated as
        control tokens unless ``allow_special`` is set (prevents prompt injection
        of e.g. ``<|im_end|>`` from file contents)."""
        if allow_special:
            return self.tk.encode(text, add_special_tokens=False).ids
        ids: list[int] = []
        for piece, is_special in self._split_specials(text):
            if is_special:
                # Encode the literal characters, splitting the token so it can't merge back.
                ids.extend(self.tk.encode(piece[:1], add_special_tokens=False).ids)
                ids.extend(self.tk.encode(piece[1:], add_special_tokens=False).ids)
            elif piece:
                ids.extend(self.tk.encode(piece, add_special_tokens=False).ids)
        return ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        if not any(tok in t for t in texts for tok in SPECIAL_TOKENS):
            return [e.ids for e in self.tk.encode_batch(list(texts), add_special_tokens=False)]
        return [self.encode(t) for t in texts]

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=skip_special)

    def _split_specials(self, text: str):
        i, start = 0, 0
        while i < len(text):
            if text[i] == "<":
                for tok in SPECIAL_TOKENS:
                    if text.startswith(tok, i):
                        yield text[start:i], False
                        yield tok, True
                        i += len(tok)
                        start = i
                        break
                else:
                    i += 1
            else:
                i += 1
        yield text[start:], False

    # ---------------------------------------------------------------- chat
    def encode_chat(self, messages: Sequence[dict], add_generation_prompt: bool = False) -> EncodedChat:
        """Encode a conversation in ChatML with an assistant-only loss mask.

        Assistant messages may contain tool-call special tokens, which are
        encoded as control tokens. All other roles are treated as untrusted text.
        """
        ids: list[int] = []
        mask: list[int] = []
        nl = self.tk.encode("\n", add_special_tokens=False).ids

        for msg in messages:
            role, content = msg["role"], msg.get("content") or ""
            if role not in ROLES:
                raise ValueError(f"unknown role {role!r}")
            header = [self.im_start_id] + self.tk.encode(role, add_special_tokens=False).ids + nl
            ids += header
            mask += [0] * len(header)
            is_assistant = role == "assistant"
            body = self.encode(content, allow_special=is_assistant) + [self.im_end_id]
            train_on = 1 if is_assistant and msg.get("train", True) else 0
            ids += body
            mask += [train_on] * len(body)
            ids += nl
            mask += [0] * len(nl)

        if add_generation_prompt:
            header = [self.im_start_id] + self.tk.encode("assistant", add_special_tokens=False).ids + nl
            ids += header
            mask += [0] * len(header)
        return EncodedChat(ids, mask)

    def encode_fim(self, prefix: str, suffix: str) -> list[int]:
        return (
            [self.special[FIM_PREFIX]]
            + self.encode(prefix)
            + [self.special[FIM_SUFFIX]]
            + self.encode(suffix)
            + [self.special[FIM_MIDDLE]]
        )


class StreamDecoder:
    """Incrementally decode token ids into text without splitting UTF-8 chars."""

    def __init__(self, tokenizer: YohoTokenizer) -> None:
        self.tok = tokenizer
        self.ids: list[int] = []
        self.emitted = 0  # characters already returned

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):  # incomplete multi-byte sequence; wait for more
            return ""
        delta = text[self.emitted :]
        self.emitted = len(text)
        return delta
