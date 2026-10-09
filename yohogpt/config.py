"""Configuration objects for YoHoGPT models and training runs.

Configs are plain dataclasses that can be loaded from / saved to YAML or JSON so
that every checkpoint carries the exact architecture it was trained with.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def _from_dict(cls, data: dict[str, Any]):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


@dataclass
class ModelConfig:
    """Decoder-only transformer (Llama-style) hyperparameters.

    The layout deliberately matches HuggingFace ``LlamaForCausalLM`` so that a
    trained checkpoint can be exported and served by vLLM / TGI / llama.cpp.
    """

    name: str = "yoho-tiny"
    vocab_size: int = 49152
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4  # grouped-query attention; == n_heads for MHA
    ffn_hidden: int | None = None  # defaults to ~8/3 * d_model rounded up
    multiple_of: int = 64
    max_seq_len: int = 2048
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.ffn_hidden is None:
            hidden = int(8 * self.d_model / 3)
            self.ffn_hidden = self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def num_params(self, include_embeddings: bool = True) -> int:
        attn = self.d_model * self.head_dim * (2 * self.n_heads + 2 * self.n_kv_heads)
        mlp = 3 * self.d_model * self.ffn_hidden
        norms = 2 * self.d_model
        total = self.n_layers * (attn + mlp + norms) + self.d_model
        if include_embeddings:
            emb = self.vocab_size * self.d_model
            total += emb if self.tie_embeddings else 2 * emb
        return total

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return _from_dict(cls, _load_mapping(path))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DataMixture:
    """One entry of a pretraining mixture: a directory of packed shards + weight."""

    name: str
    path: str
    weight: float = 1.0


@dataclass
class TrainConfig:
    """Shared configuration for pretraining, SFT and DPO runs."""

    run_name: str = "yoho-run"
    stage: str = "pretrain"  # pretrain | sft | dpo
    model: str = "configs/model/yoho-tiny.yaml"
    tokenizer: str = "artifacts/tokenizer/tokenizer.json"
    out_dir: str = "checkpoints/yoho-run"
    init_from: str | None = None  # checkpoint dir to start from (SFT/DPO/continued pretraining)

    # data
    mixture: list[DataMixture] = field(default_factory=list)  # pretrain
    sft_files: list[str] = field(default_factory=list)  # sft / dpo jsonl files
    val_fraction: float = 0.002
    seq_len: int = 2048

    # batch: global_batch_tokens = micro_batch * seq_len * grad_accum * world_size
    micro_batch: int = 4
    global_batch_tokens: int = 524_288

    # optimisation
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 100_000
    decay: str = "cosine"  # cosine | linear | wsd (warmup-stable-decay)
    wsd_decay_fraction: float = 0.2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dpo_beta: float = 0.1
    optimizer: str = "adamw"  # adamw | adamw8bit | lion | adafactor  (see yohogpt/train/optim.py)

    # system
    precision: str = "auto"  # auto | bf16 | fp16 | fp32  (auto: best the GPU supports)
    parallelism: str = "ddp"  # ddp | fsdp
    compile: bool = False  # ignored (with a warning) on GPUs that can't run Triton (< sm_70)
    grad_checkpointing: bool = False
    loss_chunk_tokens: int = 2048  # chunked cross-entropy; 0 = materialise all logits
    prefetch_batches: int = 4  # batches prepared by a background thread; 0 = off
    seed: int = 1337

    # bookkeeping
    log_interval: int = 10
    eval_interval: int = 1000
    eval_batches: int = 50
    ckpt_interval: int = 1000
    keep_checkpoints: int = 3
    wandb_project: str | None = None

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> "TrainConfig":
        data = _load_mapping(path)
        if overrides:
            data.update(overrides)
        data["mixture"] = [m if isinstance(m, DataMixture) else DataMixture(**m) for m in data.get("mixture", [])]
        return _from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` CLI overrides, interpreting values as YAML scalars."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out
