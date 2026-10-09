"""YoHoGPT transformer.

A Llama-style decoder-only transformer:
  * RMSNorm pre-normalisation
  * Rotary position embeddings (rotate-half convention, HF compatible)
  * Grouped-query attention via ``scaled_dot_product_attention`` (FlashAttention on GPU)
  * SwiGLU feed-forward
  * Optional KV cache for fast autoregressive decoding

Module/parameter names intentionally mirror ``transformers.LlamaForCausalLM`` so
``yohogpt.export_hf`` is a pure rename.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_tables(head_dim: int, max_len: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D)
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class KVCache:
    """Pre-allocated key/value cache for one attention layer."""

    def __init__(self, batch: int, max_len: int, n_kv_heads: int, head_dim: int, dtype, device) -> None:
        shape = (batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    def update(self, start: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        end = start + k.size(2)
        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v
        return self.k[:, :, :end], self.v[:, :, :end]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None, start_pos: int = 0):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(start_pos, k, v)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        mask, causal = None, False
        if T > 1:
            if start_pos == 0:
                causal = True
            else:  # chunked prefill on top of an existing cache
                mask = torch.ones(T, start_pos + T, dtype=torch.bool, device=x.device).tril(diagonal=start_pos)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache: KVCache | None = None, start_pos: int = 0):
        x = x + self.drop(self.self_attn(self.input_layernorm(x), cos, sin, cache, start_pos))
        x = x + self.drop(self.mlp(self.post_attention_layernorm(x)))
        return x


class YohoGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.grad_checkpointing = False

        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.caches: list[KVCache] | None = None
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        std = self.cfg.init_std
        resid_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------- KV cache
    def setup_cache(self, batch: int, max_len: int | None = None, dtype=None, device=None) -> None:
        max_len = max_len or self.cfg.max_seq_len
        dtype = dtype or self.embed_tokens.weight.dtype
        device = device or self.embed_tokens.weight.device
        self.caches = [
            KVCache(batch, max_len, self.cfg.n_kv_heads, self.cfg.head_dim, dtype, device) for _ in self.layers
        ]

    def clear_cache(self) -> None:
        self.caches = None

    # --------------------------------------------------------------- forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
        last_only: bool = False,
        return_logits: bool = True,
        loss_chunk: int = 0,
    ):
        """Run the model.

        Args:
            idx: (B, T) token ids.
            targets: (B, T) next-token labels; ``-100`` positions are ignored.
            start_pos: absolute position of ``idx[:, 0]`` (for KV-cached decoding).
            use_cache: read/write ``self.caches`` (call ``setup_cache`` first).
            last_only: only compute logits for the final position (decoding).
            return_logits: if False (training), compute the loss in chunks of
                ``loss_chunk`` tokens without ever materialising the full
                (tokens x vocab) logits; returns ``(None, loss)``.
        Returns:
            (logits, loss) where loss is ``None`` when ``targets`` is ``None``.
        """
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {start_pos + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        if use_cache and self.caches is None:
            raise RuntimeError("call setup_cache() before forward(use_cache=True)")

        x = self.embed_tokens(idx)
        for i, layer in enumerate(self.layers):
            cache = self.caches[i] if use_cache else None
            if self.grad_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, None, 0, use_reentrant=False)
            else:
                x = layer(x, cos, sin, cache, start_pos)
        x = self.norm(x)

        if targets is not None and not return_logits:
            return None, self.chunked_loss(x, targets, loss_chunk)
        if last_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    def _ce_sum(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.lm_head(h).float(), t, ignore_index=-100, reduction="sum")

    def chunked_loss(self, h: torch.Tensor, targets: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        """Mean next-token cross-entropy, computed ``chunk`` tokens at a time.

        Each chunk's logits are recomputed in backward (activation checkpointing), so
        peak memory is O(chunk x vocab) instead of O(batch x seq x vocab). With a
        49k vocabulary and 16k tokens per micro-batch that is ~3 GB saved.
        """
        h = h.reshape(-1, h.size(-1))
        t = targets.reshape(-1)
        chunk = chunk if chunk > 0 else h.size(0)
        total = torch.zeros((), device=h.device, dtype=torch.float32)
        for i in range(0, h.size(0), chunk):
            hc, tc = h[i : i + chunk], t[i : i + chunk]
            if torch.is_grad_enabled() and h.requires_grad:
                total = total + checkpoint(self._ce_sum, hc, tc, use_reentrant=False)
            else:
                total = total + self._ce_sum(hc, tc)
        return total / (t != -100).sum().clamp(min=1)

    # ---------------------------------------------------------- optimiser
    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str,
                            name: str = "adamw"):
        from .train.optim import make_optimizer

        return make_optimizer(name, self, lr, weight_decay, betas, device_type)

    def flops_per_token(self, seq_len: int) -> float:
        """Approximate training FLOPs per token (6N + attention term), for MFU logging."""
        n = self.num_params()
        c = self.cfg
        return 6 * n + 12 * c.n_layers * c.d_model * seq_len
