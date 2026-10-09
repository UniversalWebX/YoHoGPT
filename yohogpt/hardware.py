"""Hardware detection: GPU specs, best precision, and what features a device supports.

Peak numbers are approximate dense throughput (TFLOPS) with fp32 accumulation, as
used by autocast. They drive MFU logging and ``yoho plan`` estimates; use
``yoho benchmark`` for measured numbers on your machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUSpec:
    name: str
    fp32_tflops: float
    fp16_tflops: float  # half precision (tensor cores where present)
    bf16_tflops: float | None  # None = not supported in hardware
    memory_gb: float
    capability: tuple[int, int]

    def peak_flops(self, precision: str) -> float:
        tf = {"fp32": self.fp32_tflops, "fp16": self.fp16_tflops, "bf16": self.bf16_tflops}[precision]
        if tf is None:
            raise ValueError(f"{self.name} does not support {precision}")
        return tf * 1e12


# fmt: off
GPUS: dict[str, GPUSpec] = {g.name: g for g in [
    # Pascal (sm_61): no tensor cores, fp16 runs at 1/64 rate -> train in fp32
    GPUSpec("gtx1050",      1.86,   0.03,  None,  2,  (6, 1)),
    GPUSpec("gtx1050ti",    2.14,   0.03,  None,  4,  (6, 1)),
    GPUSpec("gtx1060",      4.4,    0.07,  None,  6,  (6, 1)),
    GPUSpec("gtx1080ti",   11.3,    0.18,  None, 11,  (6, 1)),
    # Turing GTX (sm_75): no tensor cores but double-rate fp16
    GPUSpec("gtx1650",      3.0,    6.0,   None,  4,  (7, 5)),
    GPUSpec("gtx1650super", 4.4,    8.8,   None,  4,  (7, 5)),
    GPUSpec("gtx1660super", 5.0,   10.1,   None,  6,  (7, 5)),
    # Turing RTX (sm_75): fp16 tensor cores
    GPUSpec("rtx2060",      6.5,   26.0,   None,  6,  (7, 5)),
    GPUSpec("rtx2080ti",   13.4,   53.8,   None, 11,  (7, 5)),
    GPUSpec("t4",           8.1,   65.0,   None, 16,  (7, 5)),
    # Ampere / Ada / Hopper (sm_80+): bf16 tensor cores
    GPUSpec("rtx3060",     12.7,   25.5,  25.5,  12,  (8, 6)),
    GPUSpec("rtx3090",     35.6,   71.0,  71.0,  24,  (8, 6)),
    GPUSpec("rtx4060",     15.1,   30.2,  30.2,   8,  (8, 9)),
    GPUSpec("rtx4090",     82.6,  165.2, 165.2,  24,  (8, 9)),
    GPUSpec("a100",        19.5,  312.0, 312.0,  80,  (8, 0)),
    GPUSpec("h100",        67.0,  989.0, 989.0,  80,  (9, 0)),
]}
# fmt: on


def _normalise(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum()).replace("geforce", "").replace("nvidia", "")


def lookup(name: str) -> GPUSpec:
    """Find a GPU by (loose) name, e.g. 'GTX 1050 Ti', 'NVIDIA GeForce RTX 4090', 'h100'."""
    key = _normalise(name)
    # longest match first so "gtx1650super" wins over "gtx1650"; "1050ti" matches "gtx1050ti"
    for spec_name in sorted(GPUS, key=len, reverse=True):
        core = spec_name[3:] if spec_name.startswith(("gtx", "rtx")) else spec_name
        if spec_name in key or core in key:
            return GPUS[spec_name]
    raise KeyError(f"unknown GPU {name!r}; known: {', '.join(GPUS)}")


def detect(device: torch.device | None = None) -> GPUSpec | None:
    """Spec of the current CUDA device (None on CPU / unknown GPUs)."""
    if not torch.cuda.is_available():
        return None
    idx = (device.index if device is not None and device.index is not None else torch.cuda.current_device())
    try:
        return lookup(torch.cuda.get_device_name(idx))
    except KeyError:
        return None


def precision_for_capability(cap: tuple[int, int]) -> str:
    """bf16 on Ampere+, fp16 on Volta/Turing (fast half math), fp32 on Pascal and older."""
    if cap >= (8, 0):
        return "bf16"
    if cap >= (7, 0):
        return "fp16"
    return "fp32"


def best_precision(device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    return precision_for_capability(torch.cuda.get_device_capability(device))


def supports_compile(device: torch.device) -> bool:
    """torch.compile's GPU backend (Triton) needs compute capability >= 7.0."""
    if device.type != "cuda":
        return True
    return torch.cuda.get_device_capability(device) >= (7, 0)


def peak_flops(device: torch.device, precision: str) -> float | None:
    spec = detect(device)
    if spec is None:
        return None
    try:
        return spec.peak_flops(precision)
    except ValueError:
        return None


def dtype_of(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
