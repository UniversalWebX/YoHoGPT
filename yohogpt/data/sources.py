"""Load and query the training-data catalogue (``data/sources.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CATALOGUE = Path(__file__).resolve().parents[2] / "data" / "sources.yaml"


@dataclass
class Source:
    name: str
    kind: str  # hf | git | url | local
    stage: str
    weight: float = 1.0
    license: str = "unknown"
    hf_id: str | None = None
    name_config: str | None = None
    data_dir: str | None = None
    split: str = "train"
    text_field: str = "text"
    format: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    repos: list[str] = field(default_factory=list)
    include_globs: list[str] = field(default_factory=list)
    url: str | None = None
    path: str | None = None
    gated: bool = False
    notes: str = ""


@dataclass
class Catalogue:
    sources: list[Source]
    license_allowlist: list[str]
    languages: list[str]
    eval_sets: list[dict[str, Any]]

    def stage(self, stage: str) -> list[Source]:
        return [s for s in self.sources if s.stage == stage]

    def get(self, name: str, stage: str | None = None) -> Source:
        for s in self.sources:
            if s.name == name and (stage is None or s.stage == stage):
                return s
        raise KeyError(name)

    def mixture_weights(self, stage: str) -> dict[str, float]:
        srcs = self.stage(stage)
        total = sum(s.weight for s in srcs) or 1.0
        return {s.name: s.weight / total for s in srcs}


def load_catalogue(path: str | Path = DEFAULT_CATALOGUE) -> Catalogue:
    raw = yaml.safe_load(Path(path).read_text())
    known = set(Source.__dataclass_fields__)
    sources = []
    for stage, entries in (raw.get("stages") or {}).items():
        for e in entries:
            extra = set(e) - known
            if extra:
                raise ValueError(f"source {e.get('name')}: unknown keys {sorted(extra)}")
            sources.append(Source(stage=stage, **e))
    return Catalogue(
        sources=sources,
        license_allowlist=[x.lower() for x in raw.get("license_allowlist", [])],
        languages=raw.get("languages", []),
        eval_sets=raw.get("eval", []),
    )
