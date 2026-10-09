"""Fill-in-the-middle (FIM) transformation (Bavarian et al., 2022).

Applied to a fraction of code documents during pretraining so the same model can
do left-to-right generation *and* IDE-style infilling.
"""

from __future__ import annotations

import random

from ..tokenizer.tokenizer import EOS, FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX


def fim_transform(text: str, rng: random.Random, fim_rate: float = 0.5, spm_rate: float = 0.5) -> str:
    """With probability ``fim_rate`` rearrange ``text`` into PSM or SPM order.

    PSM: <pre>prefix<suf>suffix<mid>middle
    SPM: <pre><suf>suffix<mid>prefix middle   (prefix and middle are contiguous)
    The returned string contains special-token *strings*; encode it with
    ``allow_special=True``.
    """
    if len(text) < 10 or rng.random() >= fim_rate:
        return text
    a, b = sorted(rng.randrange(len(text) + 1) for _ in range(2))
    prefix, middle, suffix = text[:a], text[a:b], text[b:]
    if rng.random() < spm_rate:
        return f"{FIM_PREFIX}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{prefix}{middle}"
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"


def strip_specials(text: str) -> str:
    """Make untrusted text safe to concatenate with special-token strings."""
    for tok in (EOS, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX):
        text = text.replace(tok, tok.replace("|", "│"))
    return text
