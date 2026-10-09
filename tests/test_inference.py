import torch

from yohogpt.config import ModelConfig
from yohogpt.inference.backends import LocalBackend, ScriptedBackend, collect
from yohogpt.inference.generate import Generator, SamplingParams, sample_next
from yohogpt.model import YohoGPT


def _gen(tokenizer):
    torch.manual_seed(0)
    m = YohoGPT(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                            max_seq_len=128))
    return Generator(m, tokenizer)


def test_greedy_is_deterministic_and_bounded(tokenizer):
    g = _gen(tokenizer)
    ids = tokenizer.encode("def add(a, b):")
    p = SamplingParams(max_new_tokens=12, temperature=0)
    a, b = g.complete(ids, p), g.complete(ids, p)
    assert a.text == b.text and a.completion_tokens <= 12
    assert a.finish_reason in ("stop", "length")


def test_prompt_longer_than_context_is_truncated(tokenizer):
    g = _gen(tokenizer)
    ids = tokenizer.encode("x = 1\n" * 200)
    out = g.complete(ids, SamplingParams(max_new_tokens=8, temperature=0.7, seed=1))
    assert out.completion_tokens <= 8


def test_stop_strings(tokenizer):
    g = _gen(tokenizer)
    ids = tokenizer.encode("print")
    full = g.complete(ids, SamplingParams(max_new_tokens=20, temperature=0)).text
    if len(full) > 4:
        stop = full[2:4]
        cut = g.complete(ids, SamplingParams(max_new_tokens=20, temperature=0, stop=[stop])).text
        assert cut == full[: full.index(stop)]


def test_sampling_filters():
    logits = torch.tensor([10.0, 9.0, -5.0, -5.0])
    gen = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, SamplingParams(temperature=1.0, top_k=2), None, gen) for _ in range(50)}
    assert picks <= {0, 1}
    picks = {sample_next(logits, SamplingParams(temperature=1.0, top_p=0.5), None, gen) for _ in range(50)}
    assert picks == {0}


def test_local_backend_chat_and_fim(tokenizer):
    be = LocalBackend.from_parts(_gen(tokenizer))
    out = collect(be.chat_stream([{"role": "user", "content": "hi"}], SamplingParams(max_new_tokens=5, temperature=0)))
    assert out.finish_reason
    out = collect(be.completion_stream("def f(", SamplingParams(max_new_tokens=5, temperature=0), suffix="\n"))
    assert out.finish_reason


def test_scripted_backend_respects_stop():
    be = ScriptedBackend(["hello <tool_call>{}</tool_call> trailing"])
    out = collect(be.chat_stream([], SamplingParams(stop=["</tool_call>"])))
    assert out.text == "hello <tool_call>{}"
